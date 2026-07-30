"""
FastAPI server for CAPTCHA solving API.

Endpoints:
    POST /solve/image   — Solve image grid CAPTCHA (hCaptcha, reCAPTCHA v2)
    POST /solve/text    — Solve text CAPTCHA (OCR)
    POST /classify      — Classify a single image
    GET  /health        — Health check
    GET  /categories    — List supported categories
"""

import logging
import time
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Solver imports — resolved relative to project root
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.clip_solver import CLIPSolver
from src.ocr_solver import OCRSolver
from src.image_utils import download_image

logger = logging.getLogger(__name__)

# --- Config ---
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config() -> dict:
    """Load config from YAML file."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


# --- Global solver instances ---
clip_solver: CLIPSolver | None = None
ocr_solver: OCRSolver | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    global clip_solver, ocr_solver

    config = load_config()
    clip_config = config.get("clip", {})
    ocr_config = config.get("ocr", {})

    logger.info("Initializing solvers...")

    clip_solver = CLIPSolver(
        model_name=clip_config.get("model", "ViT-B/32"),
        device=clip_config.get("device", None),
        threshold=clip_config.get("threshold", 0.5),
    )
    clip_solver.warmup()

    ocr_solver = OCRSolver(
        languages=ocr_config.get("languages", ["en"]),
        gpu=ocr_config.get("gpu", True),
    )

    logger.info("All solvers ready!")
    yield

    # Cleanup
    logger.info("Shutting down solvers...")
    clip_solver = None
    ocr_solver = None


# --- App ---
app = FastAPI(
    title="CAPTCHA Solver API",
    description="AI-powered CAPTCHA solver using CLIP and OCR",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/Response Models ---

class ImageSolveRequest(BaseModel):
    """Request body for image grid CAPTCHA solving."""
    image_url: str | None = Field(None, description="URL of the full grid image")
    image_base64: str | None = Field(
        None,
        description="Base64-encoded image (with or without data URI prefix)",
    )
    tile_urls: list[str] | None = Field(
        None,
        description="URLs of individual tiles (alternative to image_url)",
    )
    tile_base64: list[str] | None = Field(
        None,
        description="Base64-encoded tiles (alternative to tile_urls)",
    )
    prompt: str = Field(..., description="CAPTCHA prompt text, e.g. 'Select all buses'")
    grid: str = Field("3x3", description="Grid size: '3x3' or '4x4'")
    threshold: float | None = Field(
        None,
        description="Override confidence threshold (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    return_annotated_image: bool = Field(
        False,
        description="If true, returns a base64 encoded image with red dots on selected tiles",
    )

    def model_post_init(self, __context) -> None:
        if not any([self.image_url, self.image_base64, self.tile_urls, self.tile_base64]):
            raise ValueError(
                "Provide one of: 'image_url', 'image_base64', 'tile_urls', or 'tile_base64'"
            )


class ImageSolveResponse(BaseModel):
    """Response for image grid CAPTCHA solving."""
    selected_tiles: list[int]
    confidence: list[float]
    avg_confidence: float
    category: str
    grid: str
    solve_time_ms: float
    all_tiles: list[dict] | None = None
    annotated_image_base64: str | None = None


class TextSolveRequest(BaseModel):
    """Request body for text CAPTCHA solving."""
    image_url: str | None = Field(None, description="URL of the text CAPTCHA image")
    image_base64: str | None = Field(
        None,
        description="Base64-encoded image (with or without data URI prefix)",
    )
    preprocess: bool = Field(True, description="Apply image preprocessing")
    allowlist: str | None = Field(
        None,
        description="Allowed characters (e.g., 'abcdefghijklmnopqrstuvwxyz0123456789')",
    )

    def model_post_init(self, __context) -> None:
        if not self.image_url and not self.image_base64:
            raise ValueError("Provide either 'image_url' or 'image_base64'")


class TextSolveResponse(BaseModel):
    """Response for text CAPTCHA solving."""
    text: str
    confidence: float
    alternatives: list[dict] | None = None
    solve_time_ms: float


class ClassifyRequest(BaseModel):
    """Request body for image classification."""
    image_url: str = Field(..., description="URL of the image to classify")
    categories: list[str] | None = Field(
        None,
        description="Categories to check against. If None, checks all.",
    )


class ClassifyResponse(BaseModel):
    """Response for image classification."""
    results: dict[str, float]
    top_category: str
    top_score: float
    solve_time_ms: float


# --- Endpoints ---

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "clip_loaded": clip_solver is not None,
        "ocr_loaded": ocr_solver is not None,
        "model_info": clip_solver.get_model_info() if clip_solver else None,
    }


@app.get("/categories")
async def list_categories():
    """List all supported CAPTCHA categories."""
    if clip_solver is None:
        raise HTTPException(status_code=503, detail="CLIP solver not loaded")

    from src.prompts import CAPTCHA_PROMPTS
    return {
        "categories": list(CAPTCHA_PROMPTS.keys()),
        "total": len(CAPTCHA_PROMPTS),
    }


@app.post("/solve/image", response_model=ImageSolveResponse)
async def solve_image_captcha(request: ImageSolveRequest):
    """
    Solve an image grid CAPTCHA.

    Accepts either a full grid image URL or individual tile URLs.
    Returns which tiles to select.
    """
    if clip_solver is None:
        raise HTTPException(status_code=503, detail="CLIP solver not loaded")

    start = time.perf_counter()

    try:
        if request.tile_urls or request.tile_base64:
            # Individual tiles provided
            tile_sources = request.tile_urls or request.tile_base64
            result = clip_solver.solve_tiles(
                tile_sources=tile_sources,
                prompt=request.prompt,
                threshold=request.threshold,
                return_annotated_image=request.return_annotated_image,
            )
        else:
            # Full grid image (URL or base64)
            image_source = request.image_url or request.image_base64
            result = clip_solver.solve(
                image_source=image_source,
                prompt=request.prompt,
                grid=request.grid,
                threshold=request.threshold,
                return_annotated_image=request.return_annotated_image,
            )
    except Exception as e:
        logger.error(f"Error solving image CAPTCHA: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    elapsed_ms = (time.perf_counter() - start) * 1000
    result_dict = result.to_dict()

    return ImageSolveResponse(
        selected_tiles=result_dict["selected_tiles"],
        confidence=result_dict["confidence"],
        avg_confidence=result_dict["avg_confidence"],
        category=result_dict["category"],
        grid=result_dict["grid"],
        solve_time_ms=round(elapsed_ms, 2),
        all_tiles=result_dict["all_tiles"],
    )


@app.post("/solve/text", response_model=TextSolveResponse)
async def solve_text_captcha(request: TextSolveRequest):
    """
    Solve a text CAPTCHA using OCR.

    Returns recognized text and confidence score.
    """
    if ocr_solver is None:
        raise HTTPException(status_code=503, detail="OCR solver not loaded")

    start = time.perf_counter()

    try:
        image_source = request.image_url or request.image_base64
        result = ocr_solver.solve(
            image_source=image_source,
            preprocess=request.preprocess,
            allowlist=request.allowlist,
        )
    except Exception as e:
        logger.error(f"Error solving text CAPTCHA: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    elapsed_ms = (time.perf_counter() - start) * 1000
    result_dict = result.to_dict()

    return TextSolveResponse(
        text=result_dict["text"],
        confidence=result_dict["confidence"],
        alternatives=result_dict.get("alternatives"),
        solve_time_ms=round(elapsed_ms, 2),
    )


@app.post("/classify", response_model=ClassifyResponse)
async def classify_image(request: ClassifyRequest):
    """
    Classify a single image against CAPTCHA categories.

    Useful for debugging — shows what CLIP thinks is in the image.
    """
    if clip_solver is None:
        raise HTTPException(status_code=503, detail="CLIP solver not loaded")

    start = time.perf_counter()

    try:
        results = clip_solver.classify_image(
            image_source=request.image_url,
            categories=request.categories,
        )
    except Exception as e:
        logger.error(f"Error classifying image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    elapsed_ms = (time.perf_counter() - start) * 1000
    top_category = next(iter(results))
    top_score = results[top_category]

    return ClassifyResponse(
        results={k: round(v, 4) for k, v in results.items()},
        top_category=top_category,
        top_score=round(top_score, 4),
        solve_time_ms=round(elapsed_ms, 2),
    )


@app.post("/solve/image/upload", response_model=ImageSolveResponse)
async def solve_image_captcha_upload(
    file: UploadFile = File(..., description="Grid CAPTCHA image file"),
    prompt: str = Form("Select all matching images", description="CAPTCHA prompt text"),
    grid: str = Form("3x3", description="Grid size: '3x3' or '4x4'"),
    threshold: float | None = Form(None, description="Confidence threshold 0.0-1.0"),
    return_annotated_image: bool = Form(False, description="Return image with red dots"),
):
    """
    Solve an image grid CAPTCHA via file upload.

    Upload a screenshot of the CAPTCHA grid directly.
    """
    if clip_solver is None:
        raise HTTPException(status_code=503, detail="CLIP solver not loaded")

    start = time.perf_counter()

    try:
        image_bytes = await file.read()
        result = clip_solver.solve(
            image_source=image_bytes,
            prompt=prompt,
            grid=grid,
            threshold=threshold,
            return_annotated_image=return_annotated_image,
        )
    except Exception as e:
        logger.error(f"Error solving image CAPTCHA: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    elapsed_ms = (time.perf_counter() - start) * 1000
    result_dict = result.to_dict()

    return ImageSolveResponse(
        selected_tiles=result_dict["selected_tiles"],
        confidence=result_dict["confidence"],
        avg_confidence=result_dict["avg_confidence"],
        category=result_dict["category"],
        grid=result_dict["grid"],
        solve_time_ms=round(elapsed_ms, 2),
        all_tiles=result_dict["all_tiles"],
        annotated_image_base64=result_dict.get("annotated_image_base64"),
    )


@app.post("/solve/text/upload", response_model=TextSolveResponse)
async def solve_text_captcha_upload(
    file: UploadFile = File(..., description="Text CAPTCHA image file"),
    preprocess: bool = Form(True, description="Apply image preprocessing"),
    allowlist: str | None = Form(None, description="Allowed characters"),
):
    """
    Solve a text CAPTCHA via file upload.

    Upload a screenshot of the text CAPTCHA directly.
    """
    if ocr_solver is None:
        raise HTTPException(status_code=503, detail="OCR solver not loaded")

    start = time.perf_counter()

    try:
        image_bytes = await file.read()
        result = ocr_solver.solve(
            image_source=image_bytes,
            preprocess=preprocess,
            allowlist=allowlist,
        )
    except Exception as e:
        logger.error(f"Error solving text CAPTCHA: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    elapsed_ms = (time.perf_counter() - start) * 1000
    result_dict = result.to_dict()

    return TextSolveResponse(
        text=result_dict["text"],
        confidence=result_dict["confidence"],
        alternatives=result_dict.get("alternatives"),
        solve_time_ms=round(elapsed_ms, 2),
    )


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    config = load_config()
    server_config = config.get("server", {})

    uvicorn.run(
        "server.app:app",
        host=server_config.get("host", "0.0.0.0"),
        port=server_config.get("port", 8000),
        reload=server_config.get("reload", False),
        workers=server_config.get("workers", 1),
    )
