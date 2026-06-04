# NutriVision AI

NutriVision AI adalah aplikasi deteksi dan analisis nutrisi makanan berbasis Computer Vision dan Deep Learning untuk tugas akhir Praktikum Pengolahan Citra Digital (PCD).

Aplikasi ini dapat:
- Mendeteksi jenis makanan dari gambar
- Melakukan segmentasi objek dengan kontur dan filter Gaussian
- Menggunakan model CNN dengan Test-Time Augmentation (TTA)
- Menyajikan estimasi nutrisi berdasarkan database makanan Indonesia
- Menampilkan visualisasi Grad-CAM untuk interpretabilitas

## Fitur Utama

- Klasifikasi 13 makanan Indonesia
- Segmentasi citra dan deteksi kontur makanan
- Estimasi berat makanan dari luas kontur
- Lookup nutrisi dari `dataset/indonesian_food/nutrition.csv`
- Antarmuka pengguna grafis dengan Tkinter

## Struktur Proyek

```
.
├── dataset/
│   ├── dataset_gambar/      # dataset gambar untuk train/valid/test
│   └── indonesian_food/
│       └── nutrition.csv   # database nutrisi
├── models/
│   ├── class_indices.txt
│   └── food_model.h5       # model CNN hasil pelatihan
├── src/
│   ├── gui.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── feature_extraction.py
│   ├── color_corrector.py
│   ├── gradcam.py
│   ├── calorie.py
│   └── train_model.py
├── main.py
└── README.md
```

## Dataset & Ruang Lingkup

### Dataset 1: Indonesian Food Dataset

- **Sumber:** [kaggle.com/datasets/rizkyyk/dataset-food-classification](https://www.kaggle.com/datasets/rizkyyk/dataset-food-classification)
- **Kelas (13 makanan):** Ayam Goreng, Burger, French Fries, Gado-Gado, Ikan Goreng, Mie Goreng, Nasi Goreng, Nasi Padang, Pizza, Rawon, Rendang, Sate, dan Soto
- **Pembagian data:** 80% training dan 20% validation
- **Digunakan untuk:** training model CNN

### Dataset 2: Indonesian Food and Drink Nutrition Dataset

- **Sumber:** [kaggle.com/datasets/anasfikrihanif/indonesian-food-and-drink-nutrition-dataset](https://www.kaggle.com/datasets/anasfikrihanif/indonesian-food-and-drink-nutrition-dataset)
- **Format data:** `nutrition.csv`
- **Jumlah entri:** 1.346 entri makanan dan minuman Indonesia
- **Digunakan sebagai:** database nutrisi
- **Informasi yang tersedia:** kalori, protein, lemak, karbohidrat, gula, serat, sodium

## Metodologi

Pipeline NutriVision AI terdiri dari 7 tahap utama, dari input gambar hingga output nutrisi:

```
Input Gambar
     │
     ▼
1. Preprocessing
     │
     ▼
2. Segmentasi & Deteksi Kontur
     │
     ▼
3. Estimasi Berat
     │
     ▼
4. Klasifikasi CNN + TTA
     │
     ▼
5. Visualisasi Grad-CAM
     │
     ▼
6. Lookup Nutrisi
     │
     ▼
Output (Label + Nutrisi)
```

### 1. Preprocessing

Diterapkan dua jenis preprocessing sebelum klasifikasi:

- **Domain Spasial:** Gambar di-resize ke 300×300, dilakukan Gaussian Blur kernel 5×5 untuk meredam noise, kemudian sharpening menggunakan kernel Laplacian 3×3.
- **Domain Frekuensi:** Konversi ke grayscale → FFT 2D → Gaussian Lowpass Filter (GLPF) dengan cutoff D0=40 → Inverse FFT → normalisasi ke [0, 255].

### 2. Segmentasi & Deteksi Kontur

- Konversi gambar ke ruang warna HSV.
- Color masking untuk warna dominan makanan (kuning, oranye, coklat, merah).
- Bagian atas gambar (18%) diabaikan agar logo/teks tidak ikut tersegmentasi.
- Morphological operations: **opening** lalu **closing** dengan kernel 7×7 untuk membersihkan noise dan mengisi celah.
- Deteksi kontur menggunakan `cv2.findContours`, kemudian kontur terbaik dipilih berdasarkan kombinasi luas area dan posisi tengah gambar.

### 3. Estimasi Berat

Berat makanan diestimasi dari luas piksel kontur hasil segmentasi:

```
berat (gram) = (luas_kontur / 1000) × 6
```

### 4. Klasifikasi CNN dengan Test-Time Augmentation (TTA)

- **CLAHE** (Contrast Limited Adaptive Histogram Equalization) diterapkan di ruang warna LAB (clipLimit=2.5, tileGrid 8×8) untuk meningkatkan kontras.
- **8 varian TTA** dibuat dari gambar yang sudah di-enhance:
  1. Original (CLAHE)
  2. Flip horizontal
  3. Brightness +25
  4. Brightness -25
  5. Center crop 80%
  6. Saturasi ×1.4
  7. Rotasi +10°
  8. Rotasi -10°
- Setiap varian diprediksikan oleh model CNN (MobileNetV2-based), lalu probabilitas di-rata-rata (*soft voting*).
- **Color-profile correction** berbasis analisis HSV per kelas diterapkan pada probabilitas akhir untuk meningkatkan akurasi.

### 5. Visualisasi Grad-CAM

Grad-CAM digunakan untuk menginterpretasikan keputusan model:

- Sub-model dibuat untuk merekam output layer konvolusi terakhir (`Conv_1` / `out_relu` pada MobileNetV2).
- `GradientTape` digunakan untuk menghitung gradien kelas target terhadap output layer tersebut.
- Gradien di-pooling secara global (Global Average Pooling) → *weighted sum* terhadap feature map → ReLU → normalisasi ke [0, 1].
- Heatmap di-resize ke ukuran gambar asli, diberi colormap JET, lalu di-overlay dengan komposisi **60% gambar asli + 40% heatmap**.

### 6. Lookup Nutrisi

- Nama kelas dari klasifikasi dinormalisasi (lowercase, strip) lalu dicocokan ke `CLASS_MAPPING` yang memetakan label CNN ke entri dalam `nutrition.csv`.
- Jika ditemukan di database, nilai nutrisi diambil dari CSV; kolom yang kosong (serat, gula, sodium) diisi dari nilai fallback predefined.
- Jika tidak ditemukan di database, digunakan nilai fallback manual per kelas.
- Semua nilai nutrisi dalam CSV adalah **per 100g**, kemudian diskala proporsional terhadap estimasi berat:

```
nilai_nutrisi = nilai_per_100g × (berat_estimasi / 100)
```

### 7. Output

Hasil akhir yang ditampilkan ke pengguna:
- Label makanan dan confidence score
- Visualisasi segmentasi dengan kontur makanan
- Heatmap Grad-CAM sebagai interpretasi model
- Estimasi nutrisi lengkap: kalori, protein, lemak, karbohidrat, serat, gula, sodium

## Dependensi

Aplikasi membutuhkan Python 3.8 atau lebih baru.

Instal dependensi yang diperlukan:

```bash
pip install numpy opencv-python pillow tensorflow pandas
```

Jika Anda menggunakan virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
pip install numpy opencv-python pillow tensorflow pandas
```

## Cara Menjalankan

Jalankan aplikasi GUI dengan perintah:

```bash
python main.py
```

Setelah terbuka, klik `UPLOAD GAMBAR` untuk memilih gambar makanan, lalu klik `DETEKSI & ANALISIS` untuk menjalankan inferensi.

## Catatan Penting

- Pastikan `models/food_model.h5` dan `models/class_indices.txt` tersedia.
- Pastikan `dataset/indonesian_food/nutrition.csv` tersedia.
- Folder `dataset/dataset_gambar` berisi data gambar untuk pelatihan dan validasi.

## Anggota Tim

| NIM | Nama |
|-----|------|
| 152024073 | Danendra Yusup Arly |
| 152024166 | Edsel Sultan Farel |
| 152024177 | Candra Kurniawan |
| 152024195 | Moch Sayyid Ath Thariq |
| 152024196 | Thalibul Huda Assuja |

## Lisensi

Proyek ini dibuat untuk keperluan tugas akhir kuliah pengolahan citra digital.
