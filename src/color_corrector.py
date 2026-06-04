"""
color_corrector.py
==================
Color-profile-based correction layer for CNN food predictions.

After the CNN produces class probabilities, this module:
  1. Analyzes the actual pixel-color distribution (HSV histogram) of the image.
  2. Computes a "color compatibility score" for every food class.
  3. Blends CNN score with color score (soft fusion) → more accurate final ranking.

Each food class has:
  - expected_hue_ranges  : list of (h_low, h_high) bands (OpenCV: 0-179)
  - expected_sat_range   : (s_low, s_high) typical saturation band (0-255)
  - expected_val_range   : (v_low, v_high) typical brightness band (0-255)
  - weight               : how strongly to trust the color signal for this class

The resulting color_score is combined with the CNN probability:
  final = CNN_WEIGHT * cnn_prob + COLOR_WEIGHT * color_score
"""

import cv2
import numpy as np

# ── Weights for blending CNN vs colour signal ────────────────────────────────
CNN_WEIGHT   = 0.55   # How much we trust the CNN
COLOR_WEIGHT = 0.45   # How much we trust the colour analysis

# ── Per-class colour profiles ────────────────────────────────────────────────
# Hue in OpenCV scale (0-179), Saturation 0-255, Value 0-255
# Multiple hue bands are OR-ed together.
FOOD_COLOR_PROFILES = {
    # ── Mie Goreng ────────────────────────────────────────────────────────────
    # Yellow/golden noodles with high brightness; often has reds & greens too.
    "Mie Goreng": {
        "hue_bands": [(15, 38), (0, 12)],          # yellow-orange + slight red
        "sat_range": (60, 255),                      # well saturated
        "val_range": (100, 255),                     # bright (not dark)
        "weight": 1.0,
    },

    # ── Rendang ───────────────────────────────────────────────────────────────
    # Very dark brown/black; low brightness, warm but desaturated.
    "Rendang": {
        "hue_bands": [(0, 20), (160, 179)],         # dark reddish-brown
        "sat_range": (30, 200),
        "val_range": (10, 110),                      # dark (key difference!)
        "weight": 1.0,
    },

    # ── Ayam Goreng ───────────────────────────────────────────────────────────
    # Golden-brown crust, medium brightness.
    "Ayam Goreng": {
        "hue_bands": [(10, 30)],
        "sat_range": (50, 220),
        "val_range": (80, 200),
        "weight": 0.8,
    },

    # ── Nasi Goreng ───────────────────────────────────────────────────────────
    # Brown rice, egg yellow, mix of warm tones.
    "Nasi Goreng": {
        "hue_bands": [(10, 35)],
        "sat_range": (30, 180),
        "val_range": (80, 210),
        "weight": 0.8,
    },

    # ── Nasi Padang ───────────────────────────────────────────────────────────
    # White rice dominant + many colourful side dishes.
    "Nasi Padang": {
        "hue_bands": [(0, 179)],                    # mixed — rely on CNN more
        "sat_range": (0, 255),
        "val_range": (100, 255),
        "weight": 0.5,
    },

    # ── Gado-Gado ────────────────────────────────────────────────────────────
    # Peanut sauce (yellow-orange) + green vegetables.
    "Gado-Gado": {
        "hue_bands": [(25, 45), (35, 85)],          # yellow + green
        "sat_range": (50, 255),
        "val_range": (80, 230),
        "weight": 0.9,
    },

    # ── Ikan Goreng ───────────────────────────────────────────────────────────
    # Golden-fried fish, similar to ayam goreng but often lighter.
    "Ikan Goreng": {
        "hue_bands": [(10, 35)],
        "sat_range": (40, 200),
        "val_range": (100, 230),
        "weight": 0.8,
    },

    # ── Rawon ────────────────────────────────────────────────────────────────
    # Very dark (almost black) soup from keluak; similar darkness to rendang.
    "Rawon": {
        "hue_bands": [(0, 20)],
        "sat_range": (20, 160),
        "val_range": (5, 80),                        # very dark
        "weight": 0.9,
    },

    # ── Sate ─────────────────────────────────────────────────────────────────
    # Charred brown skewers + peanut sauce (yellow-orange).
    "Sate": {
        "hue_bands": [(5, 28), (10, 35)],
        "sat_range": (40, 220),
        "val_range": (50, 200),
        "weight": 0.7,
    },

    # ── Soto ─────────────────────────────────────────────────────────────────
    # Yellow-clear broth, lots of yellow/white tones.
    "Soto": {
        "hue_bands": [(15, 40)],
        "sat_range": (20, 180),
        "val_range": (140, 255),                     # lighter than rendang
        "weight": 0.85,
    },

    # ── Burger ───────────────────────────────────────────────────────────────
    # Brown bun + green lettuce + red tomato; medium brightness.
    "Burger": {
        "hue_bands": [(5, 30), (35, 80)],
        "sat_range": (40, 240),
        "val_range": (60, 210),
        "weight": 0.75,
    },

    # ── French Fries ─────────────────────────────────────────────────────────
    # Bright golden-yellow, very high brightness.
    "French Fries": {
        "hue_bands": [(18, 38)],
        "sat_range": (50, 230),
        "val_range": (150, 255),                     # very bright
        "weight": 0.9,
    },

    # ── Pizza ─────────────────────────────────────────────────────────────────
    # Red tomato sauce + yellow cheese; warm tones.
    "Pizza": {
        "hue_bands": [(0, 15), (15, 35)],
        "sat_range": (50, 255),
        "val_range": (100, 240),
        "weight": 0.85,
    },
}


def _compute_color_score(img_bgr: np.ndarray, profile: dict) -> float:
    """
    Returns [0.0, 1.0] representing how well an image matches a colour profile.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    total_pixels = h * w

    # ── Build combined hue mask ───────────────────────────────────────────
    hue_mask = np.zeros((h, w), dtype=np.uint8)
    for h_lo, h_hi in profile["hue_bands"]:
        lo = np.array([h_lo, 0, 0], dtype=np.uint8)
        hi = np.array([h_hi, 255, 255], dtype=np.uint8)
        hue_mask = cv2.bitwise_or(hue_mask, cv2.inRange(hsv, lo, hi))

    # ── Saturation mask ───────────────────────────────────────────────────
    s_lo, s_hi = profile["sat_range"]
    sat_mask = cv2.inRange(hsv[:, :, 1:2],
                           np.array([s_lo], dtype=np.uint8),
                           np.array([s_hi], dtype=np.uint8))

    # ── Value (brightness) mask ───────────────────────────────────────────
    v_lo, v_hi = profile["val_range"]
    val_mask = cv2.inRange(hsv[:, :, 2:3],
                           np.array([v_lo], dtype=np.uint8),
                           np.array([v_hi], dtype=np.uint8))

    # ── Combined mask ─────────────────────────────────────────────────────
    combined = cv2.bitwise_and(hue_mask, sat_mask)
    combined = cv2.bitwise_and(combined, val_mask)

    match_pixels = float(np.count_nonzero(combined))
    score = match_pixels / total_pixels
    return min(score * 3.0, 1.0)   # scale up since food rarely fills 100% pixels


def apply_color_correction(
    img_bgr: np.ndarray,
    class_probs: np.ndarray,
    idx_to_class: dict
) -> np.ndarray:
    """
    Blend CNN class probabilities with colour-profile scores.

    Parameters
    ----------
    img_bgr      : OpenCV BGR image (numpy array)
    class_probs  : 1-D numpy array of raw CNN probabilities (sums to ~1)
    idx_to_class : dict mapping class index (int) → class name (str)

    Returns
    -------
    corrected_probs : 1-D numpy array (re-normalized), same shape as class_probs
    """
    n_classes = len(class_probs)
    color_scores = np.zeros(n_classes, dtype=np.float32)

    for idx in range(n_classes):
        label = idx_to_class.get(idx, "")
        profile = FOOD_COLOR_PROFILES.get(label)
        if profile is None:
            color_scores[idx] = 0.5    # neutral if no profile defined
        else:
            raw = _compute_color_score(img_bgr, profile)
            color_scores[idx] = raw * profile.get("weight", 1.0)

    # Normalize colour scores to [0, 1]
    max_cs = color_scores.max()
    if max_cs > 0:
        color_scores /= max_cs

    # ── Soft fusion ──────────────────────────────────────────────────────
    fused = CNN_WEIGHT * class_probs + COLOR_WEIGHT * color_scores

    # Re-normalize so probabilities sum to 1
    total = fused.sum()
    if total > 0:
        fused /= total

    return fused
