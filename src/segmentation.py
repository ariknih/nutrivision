import cv2
import numpy as np


def segment_food(image):
    h, w = image.shape[:2]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Fokus warna makanan: kuning, oranye, coklat, merah
    lower1 = np.array([5, 40, 40])
    upper1 = np.array([35, 255, 255])

    lower2 = np.array([0, 50, 50])
    upper2 = np.array([10, 255, 255])

    lower3 = np.array([160, 50, 50])
    upper3 = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask3 = cv2.inRange(hsv, lower3, upper3)

    mask = mask1 + mask2 + mask3

    # Abaikan bagian atas gambar agar logo/tulisan tidak ikut
    mask[0:int(h * 0.18), :] = 0

    kernel = np.ones((7, 7), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    object_mask = np.zeros(mask.shape, dtype=np.uint8)

    if len(contours) == 0:
        return image, object_mask, 0

    candidates = []

    image_area = h * w

    for c in contours:
        area = cv2.contourArea(c)

        if area < 800:
            continue

        x, y, bw, bh = cv2.boundingRect(c)

        box_area = bw * bh
        ratio = box_area / image_area

        # Hindari bounding box yang hampir memenuhi seluruh gambar (diperlonggar dari 0.65 ke 0.85)
        if ratio > 0.85:
            continue

        # Hindari objek terlalu kecil
        if ratio < 0.01:
            continue

        # Prioritaskan area tengah/bawah gambar
        center_y = y + bh / 2
        center_score = center_y / h

        score = area * center_score

        candidates.append((score, c))

    if len(candidates) == 0:
        largest = max(contours, key=cv2.contourArea)
    else:
        largest = max(candidates, key=lambda x: x[0])[1]

    area = int(cv2.contourArea(largest))

    cv2.drawContours(
        object_mask,
        [largest],
        -1,
        255,
        -1
    )

    segmented = cv2.bitwise_and(
        image,
        image,
        mask=object_mask
    )

    return segmented, object_mask, area