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

## Lisensi

Proyek ini dibuat untuk keperluan tugas akhir kuliah pengolahan citra digital.
