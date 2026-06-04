"""
predict.py
==========
Food image classification with:
  1. CLAHE contrast enhancement
  2. Test-Time Augmentation (TTA) — 6 variants, soft-voting
  3. Color-profile correction (HSV analysis per class)

The pipeline:
  raw image → CLAHE → TTA variants → CNN → avg probs
           → Color-profile scorer → fused probs → top-1 / top-k
"""

import os
import ast
import numpy as np
import cv2

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image

MODEL_PATH = "models/food_model.h5"
CLASS_INDICES_PATH = "models/class_indices.txt"

# ── Singleton model cache ─────────────────────────────────────────────────────
_model = None
_idx_to_class = None


def _load_model():
    global _model, _idx_to_class
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
    if _idx_to_class is None:
        with open(CLASS_INDICES_PATH, "r") as f:
            class_indices = ast.literal_eval(f.read())
        _idx_to_class = {v: k for k, v in class_indices.items()}
    return _model, _idx_to_class


# ── Pre-processing helpers ────────────────────────────────────────────────────

def _clahe_enhance(img_bgr: np.ndarray) -> np.ndarray:
    """Apply CLAHE contrast enhancement in LAB space."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return enhanced


def _to_input_array(img_bgr: np.ndarray, target_size: tuple) -> np.ndarray:
    """Resize + normalize to model input tensor shape (1, H, W, 3)."""
    resized = cv2.resize(img_bgr, target_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    return np.expand_dims(rgb, axis=0)


def _tta_variants(img_bgr: np.ndarray, target_size: tuple) -> list:
    """
    Build Test-Time Augmentation variants.
    Returns list of (1, H, W, 3) float32 arrays.
    """
    enhanced = _clahe_enhance(img_bgr)
    h, w = enhanced.shape[:2]
    variants = []

    # 1. Original (CLAHE)
    variants.append(_to_input_array(enhanced, target_size))

    # 2. Horizontal flip
    variants.append(_to_input_array(cv2.flip(enhanced, 1), target_size))

    # 3. Brightness +25
    variants.append(_to_input_array(
        cv2.convertScaleAbs(enhanced, alpha=1.0, beta=25), target_size))

    # 4. Brightness −25
    variants.append(_to_input_array(
        cv2.convertScaleAbs(enhanced, alpha=1.0, beta=-25), target_size))

    # 5. Center crop 80%
    cx, cy = w // 2, h // 2
    cw, ch = int(w * 0.8), int(h * 0.8)
    crop = enhanced[max(cy - ch//2, 0):min(cy + ch//2, h),
                    max(cx - cw//2, 0):min(cx + cw//2, w)]
    variants.append(_to_input_array(crop, target_size))

    # 6. Saturation boost ×1.4
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype("float32")
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255)
    sat_img = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    variants.append(_to_input_array(sat_img, target_size))

    # 7. Slight rotation +10°
    M = cv2.getRotationMatrix2D((w//2, h//2), 10, 1.0)
    rot = cv2.warpAffine(enhanced, M, (w, h))
    variants.append(_to_input_array(rot, target_size))

    # 8. Slight rotation −10°
    M2 = cv2.getRotationMatrix2D((w//2, h//2), -10, 1.0)
    rot2 = cv2.warpAffine(enhanced, M2, (w, h))
    variants.append(_to_input_array(rot2, target_size))

    return variants


# ── Main prediction function ─────────────────────────────────────────────────

def predict_food(img_path: str):
    """
    Predict food class from image file path.

    Pipeline:
      CLAHE → 8 TTA variants → CNN soft-voting → color-profile correction

    Returns
    -------
    label      : str   — food class name (Title Case)
    confidence : float — final corrected confidence [0, 1]
    """
    model, idx_to_class = _load_model()

    # Determine model input resolution
    try:
        _, mh, mw, _ = model.input_shape
        target_size = (mw or 128, mh or 128)
    except Exception:
        target_size = (128, 128)

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Tidak bisa membaca gambar: {img_path}")

    # ── Step 1: TTA + CNN ────────────────────────────────────────────────────
    variants = _tta_variants(img_bgr, target_size)
    all_preds = []
    for inp in variants:
        try:
            pred = model.predict(inp, verbose=0)
            all_preds.append(pred[0])
        except Exception:
            pass

    if not all_preds:
        raise RuntimeError("Semua TTA variant gagal diprediksi.")

    avg_pred = np.mean(np.stack(all_preds, axis=0), axis=0)

    # ── Step 2: Color-profile correction ─────────────────────────────────────
    try:
        from color_corrector import apply_color_correction
        corrected = apply_color_correction(img_bgr, avg_pred, idx_to_class)
    except Exception as e:
        print(f"[WARN] Color correction skipped: {e}")
        corrected = avg_pred

    class_id = int(np.argmax(corrected))
    confidence = float(corrected[class_id])
    label = idx_to_class.get(class_id, "unknown")

    return label, confidence


def predict_top_k(img_path: str, k: int = 3):
    """
    Like predict_food() but returns the top-k predictions with corrected scores.

    Returns
    -------
    List of (label: str, confidence: float) tuples, sorted highest first.
    """
    model, idx_to_class = _load_model()

    try:
        _, mh, mw, _ = model.input_shape
        target_size = (mw or 128, mh or 128)
    except Exception:
        target_size = (128, 128)

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Tidak bisa membaca gambar: {img_path}")

    variants = _tta_variants(img_bgr, target_size)
    all_preds = []
    for inp in variants:
        try:
            pred = model.predict(inp, verbose=0)
            all_preds.append(pred[0])
        except Exception:
            pass

    if not all_preds:
        raise RuntimeError("Semua TTA variant gagal diprediksi.")

    avg_pred = np.mean(np.stack(all_preds, axis=0), axis=0)

    try:
        from color_corrector import apply_color_correction
        corrected = apply_color_correction(img_bgr, avg_pred, idx_to_class)
    except Exception as e:
        print(f"[WARN] Color correction skipped: {e}")
        corrected = avg_pred

    top_ids = np.argsort(corrected)[::-1][:k]
    return [(idx_to_class[i], float(corrected[i])) for i in top_ids]