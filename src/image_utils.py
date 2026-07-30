"""
Image processing utilities for CAPTCHA solver.

Handles image downloading, grid splitting, preprocessing, and transformations.
"""

import io
import logging
from pathlib import Path

import httpx
from PIL import Image, ImageFilter, ImageOps

logger = logging.getLogger(__name__)


async def download_image(url: str, timeout: float = 10.0) -> Image.Image:
    """
    Download an image from URL and return as PIL Image.

    Args:
        url: Image URL to download
        timeout: Request timeout in seconds

    Returns:
        PIL Image in RGB mode
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        return image.convert("RGB")


def download_image_sync(url: str, timeout: float = 10.0) -> Image.Image:
    """Synchronous version of download_image."""
    with httpx.Client(timeout=timeout) as client:
        response = client.get(url, follow_redirects=True)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        return image.convert("RGB")


def load_image(source: str | Path | bytes | Image.Image) -> Image.Image:
    """
    Load image from various sources: file path, URL, bytes, or PIL Image.

    Args:
        source: Image source — file path, URL string, raw bytes, or PIL Image

    Returns:
        PIL Image in RGB mode
    """
    if isinstance(source, Image.Image):
        return source.convert("RGB")

    if isinstance(source, bytes):
        return Image.open(io.BytesIO(source)).convert("RGB")

    if isinstance(source, Path):
        return Image.open(source).convert("RGB")

    if isinstance(source, str):
        if source.startswith(("http://", "https://")):
            return download_image_sync(source)
        return Image.open(source).convert("RGB")

    raise ValueError(f"Unsupported image source type: {type(source)}")


def split_grid(
    image: Image.Image,
    rows: int = 3,
    cols: int = 3,
) -> list[Image.Image]:
    """
    Split a CAPTCHA grid image into individual tiles.

    Args:
        image: Full grid image
        rows: Number of rows in grid
        cols: Number of columns in grid

    Returns:
        List of tile images, ordered left-to-right, top-to-bottom.
        Index 0 = top-left, Index (rows*cols-1) = bottom-right.
    """
    width, height = image.size
    tile_w = width // cols
    tile_h = height // rows

    tiles = []
    for row in range(rows):
        for col in range(cols):
            left = col * tile_w
            upper = row * tile_h
            right = left + tile_w
            lower = upper + tile_h
            tile = image.crop((left, upper, right, lower))
            tiles.append(tile)

    logger.debug(
        f"Split {width}x{height} image into {rows}x{cols} grid "
        f"({len(tiles)} tiles, each {tile_w}x{tile_h})"
    )
    return tiles


def preprocess_for_clip(
    image: Image.Image,
    target_size: int = 224,
) -> Image.Image:
    """
    Preprocess image for CLIP model input.
    CLIP expects 224x224 images; this resizes while maintaining aspect ratio
    and adds padding if needed.

    Args:
        image: Input image
        target_size: Target dimension (CLIP uses 224)

    Returns:
        Preprocessed image
    """
    # Resize maintaining aspect ratio, then pad to square
    image = image.convert("RGB")
    image = ImageOps.fit(image, (target_size, target_size), method=Image.LANCZOS)
    return image


def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """
    Preprocess image for OCR text recognition.
    Applies binarization, denoising, and contrast enhancement.

    Args:
        image: Raw CAPTCHA text image

    Returns:
        Preprocessed image ready for OCR
    """
    # Convert to grayscale
    img = image.convert("L")

    # Resize to larger size for better OCR
    width, height = img.size
    scale_factor = max(1, 300 // height)  # Target ~300px height
    if scale_factor > 1:
        img = img.resize(
            (width * scale_factor, height * scale_factor),
            Image.LANCZOS,
        )

    # Increase contrast
    img = ImageOps.autocontrast(img, cutoff=5)

    # Denoise with median filter
    img = img.filter(ImageFilter.MedianFilter(size=3))

    # Binarize using adaptive threshold (Otsu-like)
    # PIL doesn't have Otsu, so we use a simple threshold
    threshold = _otsu_threshold(img)
    img = img.point(lambda p: 255 if p > threshold else 0, mode="1")

    return img.convert("RGB")


def _otsu_threshold(image: Image.Image) -> int:
    """
    Compute Otsu's threshold for a grayscale PIL image.
    """
    histogram = image.histogram()
    total = sum(histogram)

    sum_total = sum(i * h for i, h in enumerate(histogram))
    sum_bg = 0.0
    weight_bg = 0
    max_variance = 0.0
    best_threshold = 0

    for t in range(256):
        weight_bg += histogram[t]
        if weight_bg == 0:
            continue

        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += t * histogram[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg

        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

        if variance > max_variance:
            max_variance = variance
            best_threshold = t

    return best_threshold


def image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """Convert PIL Image to bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


def save_tiles_debug(
    tiles: list[Image.Image],
    output_dir: str | Path,
    prefix: str = "tile",
) -> list[Path]:
    """
    Save tiles to disk for debugging.

    Args:
        tiles: List of tile images
        output_dir: Directory to save tiles
        prefix: Filename prefix

    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for i, tile in enumerate(tiles):
        path = output_dir / f"{prefix}_{i:02d}.png"
        tile.save(path)
        paths.append(path)

    logger.info(f"Saved {len(paths)} tiles to {output_dir}")
    return paths
