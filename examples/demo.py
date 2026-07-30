"""
Demo script for CAPTCHA solver.

Shows how to use the solver programmatically.
Run on a machine with GPU for best performance.
"""

import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)

# ============================================================
# Example 1: Solve an image grid CAPTCHA
# ============================================================

def demo_image_captcha():
    """Demonstrate image grid CAPTCHA solving with CLIP."""
    from src.clip_solver import CLIPSolver

    # Initialize solver
    solver = CLIPSolver(
        model_name="ViT-B/32",  # Use ViT-L/14 for better accuracy
        threshold=0.5,
    )

    print("\n" + "=" * 60)
    print("IMAGE GRID CAPTCHA SOLVER")
    print("=" * 60)

    # --- Option A: Solve from a grid image URL ---
    # result = solver.solve(
    #     image_source="https://example.com/captcha_grid.png",
    #     prompt="Select all images with a bus",
    #     grid="3x3",
    # )

    # --- Option B: Solve from a local file ---
    # result = solver.solve(
    #     image_source="/path/to/captcha_grid.png",
    #     prompt="Select all images with a bus",
    #     grid="3x3",
    # )

    # --- Option C: Solve from individual tile URLs (hCaptcha style) ---
    # result = solver.solve_tiles(
    #     tile_sources=[
    #         "https://example.com/tile_0.png",
    #         "https://example.com/tile_1.png",
    #         # ... 9 tiles for 3x3
    #     ],
    #     prompt="Please click each image containing a bus",
    # )

    # --- Demo: Classify a single image ---
    # Shows what CLIP thinks is in the image (useful for debugging)
    print("\nModel info:")
    print(solver.get_model_info())

    # Warmup
    solver.warmup()
    print("\nSolver is ready!")
    print("Uncomment one of the solve options above to test with real images.")

    # --- How to use results ---
    # print(f"\nSelected tiles: {result.selected_indices}")
    # print(f"Confidence scores: {result.confidence_scores}")
    # print(f"Average confidence: {result.avg_confidence:.3f}")
    # print(f"\nFull result dict:")
    # import json
    # print(json.dumps(result.to_dict(), indent=2))


# ============================================================
# Example 2: Solve a text CAPTCHA
# ============================================================

def demo_text_captcha():
    """Demonstrate text CAPTCHA solving with OCR."""
    from src.ocr_solver import OCRSolver

    print("\n" + "=" * 60)
    print("TEXT CAPTCHA SOLVER")
    print("=" * 60)

    # Initialize solver
    solver = OCRSolver(
        languages=["en"],
        gpu=True,
    )

    # --- Solve from URL ---
    # result = solver.solve(
    #     image_source="https://example.com/text_captcha.png",
    #     preprocess=True,
    #     allowlist="abcdefghijklmnopqrstuvwxyz0123456789",
    # )
    # print(f"Recognized text: {result.text}")
    # print(f"Confidence: {result.confidence:.3f}")

    print("\nOCR solver ready!")
    print("Uncomment the solve call above to test with real images.")


# ============================================================
# Example 3: API Client
# ============================================================

def demo_api_client():
    """Show how to call the API server."""
    print("\n" + "=" * 60)
    print("API CLIENT EXAMPLE")
    print("=" * 60)

    print("""
# Start the server first:
#   python -m server.app
# or:
#   uvicorn server.app:app --host 0.0.0.0 --port 8000

import httpx

# Solve image CAPTCHA
response = httpx.post("http://localhost:8000/solve/image", json={
    "image_url": "https://example.com/captcha_grid.png",
    "prompt": "Select all images with a bus",
    "grid": "3x3",
    "threshold": 0.5,
})
print(response.json())
# {
#   "selected_tiles": [0, 3, 6],
#   "confidence": [0.85, 0.72, 0.91],
#   "avg_confidence": 0.827,
#   "category": "bus",
#   "grid": "3x3",
#   "solve_time_ms": 123.45
# }

# Solve with individual tile URLs (hCaptcha)
response = httpx.post("http://localhost:8000/solve/image", json={
    "tile_urls": [
        "https://example.com/tile0.png",
        "https://example.com/tile1.png",
        "https://example.com/tile2.png",
        # ... 9 tiles
    ],
    "prompt": "Please click each image containing a bus",
})

# Solve text CAPTCHA
response = httpx.post("http://localhost:8000/solve/text", json={
    "image_url": "https://example.com/text_captcha.png",
    "preprocess": True,
    "allowlist": "abcdefghijklmnopqrstuvwxyz0123456789",
})
print(response.json())
# {"text": "abc123", "confidence": 0.95}

# Classify image
response = httpx.post("http://localhost:8000/classify", json={
    "image_url": "https://example.com/some_image.png",
})
print(response.json())
# {"results": {"bus": 0.92, "car": 0.31, ...}, "top_category": "bus"}
""")


# ============================================================

if __name__ == "__main__":
    import sys

    demos = {
        "image": demo_image_captcha,
        "text": demo_text_captcha,
        "api": demo_api_client,
        "all": lambda: (demo_image_captcha(), demo_text_captcha(), demo_api_client()),
    }

    if len(sys.argv) > 1 and sys.argv[1] in demos:
        demos[sys.argv[1]]()
    else:
        print("CAPTCHA Solver Demo")
        print(f"Usage: python {sys.argv[0]} [{'|'.join(demos.keys())}]")
        print("\nRunning all demos...\n")
        demo_image_captcha()
        demo_api_client()
