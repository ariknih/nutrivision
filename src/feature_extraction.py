import cv2
from skimage.feature import graycomatrix
from skimage.feature import graycoprops


def extract_features(image, mask):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    mean_color = cv2.mean(
        image,
        mask=mask
    )[:3]

    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(
        glcm,
        'contrast'
    )[0, 0]

    energy = graycoprops(
        glcm,
        'energy'
    )[0, 0]

    homogeneity = graycoprops(
        glcm,
        'homogeneity'
    )[0, 0]

    return {
        "mean_blue": round(mean_color[0], 2),
        "mean_green": round(mean_color[1], 2),
        "mean_red": round(mean_color[2], 2),
        "contrast": round(contrast, 2),
        "energy": round(energy, 2),
        "homogeneity": round(homogeneity, 2)
    }