"""
CLIP prompt templates for different CAPTCHA categories.

Each category has positive prompts (what we're looking for)
and negative prompts (what we're NOT looking for) to improve accuracy.
"""

# Main categories encountered in hCaptcha / reCAPTCHA v2
# Each key maps to a dict with "positive" and "negative" prompt lists.
# CLIP works best with diverse phrasing — more prompts = better accuracy.

CAPTCHA_PROMPTS: dict[str, dict[str, list[str]]] = {
    # --- Vehicles ---
    "bus": {
        "positive": [
            "a photo of a bus",
            "a photo of a city bus",
            "a photo of a school bus",
            "a photo of a public transit bus",
            "a bus on the road",
        ],
        "negative": [
            "a photo of a car",
            "a photo of a truck",
            "a photo of a street without a bus",
            "a photo of a building",
        ],
    },
    "car": {
        "positive": [
            "a photo of a car",
            "a photo of an automobile",
            "a sedan on the road",
            "a photo of a vehicle",
        ],
        "negative": [
            "a photo of a bus",
            "a photo of a truck",
            "a photo of a bicycle",
            "a photo of a street",
        ],
    },
    "motorcycle": {
        "positive": [
            "a photo of a motorcycle",
            "a photo of a motorbike",
            "a motorcycle on the road",
            "a photo of a scooter",
        ],
        "negative": [
            "a photo of a bicycle",
            "a photo of a car",
            "a photo of a street",
        ],
    },
    "bicycle": {
        "positive": [
            "a photo of a bicycle",
            "a photo of a bike",
            "a bicycle parked on the street",
        ],
        "negative": [
            "a photo of a motorcycle",
            "a photo of a car",
            "a photo of a person walking",
        ],
    },
    "truck": {
        "positive": [
            "a photo of a truck",
            "a photo of a pickup truck",
            "a photo of a delivery truck",
            "a large truck on the road",
        ],
        "negative": [
            "a photo of a car",
            "a photo of a bus",
            "a photo of a van",
        ],
    },
    "boat": {
        "positive": [
            "a photo of a boat",
            "a photo of a ship",
            "a boat on the water",
            "a sailboat",
            "a motorboat",
        ],
        "negative": [
            "a photo of the ocean without a boat",
            "a photo of a beach",
            "a photo of water",
        ],
    },
    "airplane": {
        "positive": [
            "a photo of an airplane",
            "a photo of a plane",
            "an airplane in the sky",
            "a jet aircraft",
        ],
        "negative": [
            "a photo of a bird",
            "a photo of the sky",
            "a photo of clouds",
        ],
    },
    "train": {
        "positive": [
            "a photo of a train",
            "a photo of a railway train",
            "a train on the tracks",
            "a locomotive",
        ],
        "negative": [
            "a photo of a bus",
            "a photo of train tracks without a train",
            "a photo of a station",
        ],
    },

    # --- Infrastructure ---
    "traffic_light": {
        "positive": [
            "a photo of a traffic light",
            "a traffic signal",
            "a photo of a stoplight",
            "traffic lights at an intersection",
        ],
        "negative": [
            "a photo of a street lamp",
            "a photo of a pole",
            "a photo of a sign",
        ],
    },
    "crosswalk": {
        "positive": [
            "a photo of a crosswalk",
            "a pedestrian crossing",
            "a zebra crossing on the road",
            "white stripes on the road for crossing",
        ],
        "negative": [
            "a photo of a road without markings",
            "a photo of a sidewalk",
            "a photo of a street",
        ],
    },
    "fire_hydrant": {
        "positive": [
            "a photo of a fire hydrant",
            "a red fire hydrant on the sidewalk",
            "a yellow fire hydrant",
        ],
        "negative": [
            "a photo of a pole",
            "a photo of a bollard",
            "a photo of a mailbox",
        ],
    },
    "parking_meter": {
        "positive": [
            "a photo of a parking meter",
            "a parking meter on the street",
        ],
        "negative": [
            "a photo of a pole",
            "a photo of a sign post",
        ],
    },
    "bridge": {
        "positive": [
            "a photo of a bridge",
            "a bridge over water",
            "a suspension bridge",
            "a pedestrian bridge",
        ],
        "negative": [
            "a photo of a road",
            "a photo of a building",
            "a photo of a river without a bridge",
        ],
    },
    "chimney": {
        "positive": [
            "a photo of a chimney",
            "a chimney on a roof",
            "a brick chimney",
            "a smokestack",
        ],
        "negative": [
            "a photo of a roof without a chimney",
            "a photo of a building",
            "a photo of a tower",
        ],
    },
    "stairs": {
        "positive": [
            "a photo of stairs",
            "a photo of a staircase",
            "steps going up",
            "a stairway",
        ],
        "negative": [
            "a photo of a ramp",
            "a photo of a hallway",
            "a photo of a floor",
        ],
    },

    # --- Nature & Animals ---
    "palm_tree": {
        "positive": [
            "a photo of a palm tree",
            "a tropical palm tree",
            "palm trees on the beach",
        ],
        "negative": [
            "a photo of a regular tree",
            "a photo of bushes",
            "a photo of a forest",
        ],
    },
    "mountain": {
        "positive": [
            "a photo of a mountain",
            "a mountain landscape",
            "a snow-capped mountain",
            "a mountain range",
        ],
        "negative": [
            "a photo of a hill",
            "a photo of a flat landscape",
            "a photo of a valley",
        ],
    },
    "river": {
        "positive": [
            "a photo of a river",
            "a river flowing through a landscape",
            "a stream of water",
        ],
        "negative": [
            "a photo of a lake",
            "a photo of the ocean",
            "a photo of a pond",
        ],
    },

    # --- Objects ---
    "stop_sign": {
        "positive": [
            "a photo of a stop sign",
            "a red stop sign",
            "a stop sign at an intersection",
        ],
        "negative": [
            "a photo of a street sign",
            "a photo of a speed limit sign",
            "a photo of a yield sign",
        ],
    },
    "mailbox": {
        "positive": [
            "a photo of a mailbox",
            "a blue mailbox",
            "a mailbox on the street",
        ],
        "negative": [
            "a photo of a trash can",
            "a photo of a box",
            "a photo of a fire hydrant",
        ],
    },

    # --- hCaptcha specific ---
    "seaplane": {
        "positive": [
            "a photo of a seaplane",
            "a seaplane on water",
            "a float plane",
            "a plane landing on water",
        ],
        "negative": [
            "a photo of a boat",
            "a photo of a regular airplane",
            "a photo of water",
        ],
    },
    "lion": {
        "positive": [
            "a photo of a lion",
            "a male lion with a mane",
            "a lion in the wild",
            "a lioness",
        ],
        "negative": [
            "a photo of a tiger",
            "a photo of a cat",
            "a photo of a dog",
        ],
    },
    "elephant": {
        "positive": [
            "a photo of an elephant",
            "an African elephant",
            "an elephant in the wild",
        ],
        "negative": [
            "a photo of a rhinoceros",
            "a photo of a hippo",
            "a photo of a large animal",
        ],
    },
    "horse": {
        "positive": [
            "a photo of a horse",
            "a horse in a field",
            "a horse running",
        ],
        "negative": [
            "a photo of a donkey",
            "a photo of a cow",
            "a photo of a dog",
        ],
    },
    "dog": {
        "positive": [
            "a photo of a dog",
            "a puppy",
            "a dog sitting",
            "a dog playing",
        ],
        "negative": [
            "a photo of a cat",
            "a photo of a wolf",
            "a photo of an animal",
        ],
    },
    "cat": {
        "positive": [
            "a photo of a cat",
            "a kitten",
            "a cat sitting",
            "a domestic cat",
        ],
        "negative": [
            "a photo of a dog",
            "a photo of a rabbit",
            "a photo of a small animal",
        ],
    },
}

# Mapping from common CAPTCHA text prompts to our category keys
# hCaptcha / reCAPTCHA prompts vary — this maps them to our categories
PROMPT_ALIASES: dict[str, str] = {
    # English
    "select all images with a bus": "bus",
    "select all images containing a bus": "bus",
    "click on all images containing a bus": "bus",
    "select all squares with buses": "bus",
    "select all images with buses": "bus",
    "select all images with a car": "car",
    "select all images with cars": "car",
    "select all images with a motorcycle": "motorcycle",
    "select all images with motorcycles": "motorcycle",
    "select all images with a bicycle": "bicycle",
    "select all images with bicycles": "bicycle",
    "select all images with a truck": "truck",
    "select all images with trucks": "truck",
    "select all images with a boat": "boat",
    "select all images with boats": "boat",
    "select all images with an airplane": "airplane",
    "select all images with a train": "train",
    "select all images with traffic lights": "traffic_light",
    "select all images with a traffic light": "traffic_light",
    "select all squares with traffic lights": "traffic_light",
    "select all images with crosswalks": "crosswalk",
    "select all images with a crosswalk": "crosswalk",
    "select all squares with crosswalks": "crosswalk",
    "select all images with a fire hydrant": "fire_hydrant",
    "select all images with fire hydrants": "fire_hydrant",
    "select all squares with fire hydrants": "fire_hydrant",
    "select all images with parking meters": "parking_meter",
    "select all images with a parking meter": "parking_meter",
    "select all images with a bridge": "bridge",
    "select all images with bridges": "bridge",
    "select all images with a chimney": "chimney",
    "select all images with chimneys": "chimney",
    "select all images with stairs": "stairs",
    "select all images with a staircase": "stairs",
    "select all images with palm trees": "palm_tree",
    "select all images with a palm tree": "palm_tree",
    "select all images with mountains": "mountain",
    "select all images with a mountain": "mountain",
    "select all images with a stop sign": "stop_sign",
    "select all squares with stop signs": "stop_sign",
    "select all images with a mailbox": "mailbox",
    # hCaptcha specific
    "please click each image containing a seaplane": "seaplane",
    "please click each image containing a lion": "lion",
    "please click each image containing an elephant": "elephant",
    "please click each image containing a horse": "horse",
    "please click each image containing a dog": "dog",
    "please click each image containing a cat": "cat",
    "please click each image containing a bus": "bus",
    "please click each image containing a boat": "boat",
    "please click each image containing a motorcycle": "motorcycle",
    "please click each image containing a bicycle": "bicycle",
    "please click each image containing a truck": "truck",
    "please click each image containing a train": "train",
    "please click each image containing a bridge": "bridge",
    "please click each image containing a chimney": "chimney",
    "please click each image containing stairs": "stairs",
}


def resolve_prompt(raw_prompt: str) -> str | None:
    """
    Resolve a raw CAPTCHA prompt text to a category key.
    Returns None if no match is found.
    """
    normalized = raw_prompt.strip().lower()

    # Direct match
    if normalized in PROMPT_ALIASES:
        return PROMPT_ALIASES[normalized]

    # Fuzzy match: check if any alias is a substring
    for alias, category in PROMPT_ALIASES.items():
        if alias in normalized or normalized in alias:
            return category

    # Try to extract a keyword from known categories
    for category in CAPTCHA_PROMPTS:
        # Convert category key to readable form: "traffic_light" -> "traffic light"
        readable = category.replace("_", " ")
        if readable in normalized:
            return category

    return None


def get_prompts(category: str) -> dict[str, list[str]]:
    """
    Get positive and negative prompts for a category.
    Returns default prompts if category not found.
    """
    if category in CAPTCHA_PROMPTS:
        return CAPTCHA_PROMPTS[category]

    # Fallback: generate generic prompts from category name
    readable = category.replace("_", " ")
    return {
        "positive": [
            f"a photo of a {readable}",
            f"a {readable}",
            f"an image containing a {readable}",
        ],
        "negative": [
            "a photo of something else",
            "an empty photo",
            "a photo without any notable objects",
        ],
    }
