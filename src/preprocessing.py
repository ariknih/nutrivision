import cv2
import numpy as np


def spatial_preprocessing(image):

    resized = cv2.resize(image, (300, 300))

    blur = cv2.GaussianBlur(
        resized,
        (5, 5),
        0
    )

    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    sharpen = cv2.filter2D(
        blur,
        -1,
        sharpen_kernel
    )

    return sharpen


def frequency_preprocessing(image):
    """
    Gaussian Lowpass Filter (GLPF) di Domain Frekuensi:
    1. Konversi ke Grayscale & resize ke 300x300.
    2. Transformasi Fourier 2D (FFT) & Shift.
    3. Buat mask Gaussian Lowpass H(u,v) = exp(-D^2 / (2 * D0^2)) dengan D0 = 40.
    4. Kalikan spektrum frekuensi dengan mask H.
    5. Inverse FFT (IFFT) & Shift balik.
    6. Ambil magnitude absolut & normalisasi ke [0, 255].
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (300, 300))

    # FFT 2D
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    # Buat Gaussian Lowpass Filter mask
    h, w = gray.shape
    cx, cy = w // 2, h // 2
    u = np.arange(w)
    v = np.arange(h)
    U, V = np.meshgrid(u - cx, v - cy)
    D = np.sqrt(U**2 + V**2)
    
    # Cutoff frequency D0 = 40
    d0 = 40.0
    H = np.exp(-(D**2) / (2 * (d0**2)))

    # Fusi filter di domain frekuensi
    gshift = fshift * H

    # Inverse FFT
    g = np.fft.ifftshift(gshift)
    img_back = np.fft.ifft2(g)
    img_back = np.abs(img_back)

    # Normalisasi hasil filter ke rentang [0, 255]
    img_back = np.uint8(cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX))

    return img_back