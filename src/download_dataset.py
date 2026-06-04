import kagglehub
import shutil
import os
import time


def download_dataset():

    print("=" * 50)
    print("INDONESIAN FOOD DATASET DOWNLOADER")
    print("=" * 50)

    try:
        print("\n[1] START DOWNLOAD DATASET...")
        print("Mohon tunggu, proses bisa beberapa menit...\n")

        start_time = time.time()

        # Download dataset dari Kaggle
        path = kagglehub.dataset_download(
            "anasfikrihanif/indonesian-food-and-drink-nutrition-dataset"
        )

        end_time = time.time()

        print("\n[2] DOWNLOAD SELESAI")
        print("Lokasi cache dataset:")
        print(path)

        print("\nWaktu download:",
              round(end_time - start_time, 2),
              "detik")

        # Folder tujuan project
        target = "dataset/indonesian_food"

        print("\n[3] MENYIAPKAN FOLDER DATASET...")

        # Hapus folder lama jika ada
        if os.path.exists(target):
            print("Folder lama ditemukan -> menghapus...")
            shutil.rmtree(target)

        # Buat folder dataset
        os.makedirs("dataset", exist_ok=True)

        print("\n[4] COPY DATASET KE PROJECT...")

        shutil.copytree(path, target)

        print("\n[5] DATASET BERHASIL DISIMPAN")
        print("Lokasi akhir dataset:")
        print(target)

        print("\nSELESAI!")

    except Exception as e:

        print("\n[ERROR]")
        print(str(e))

        print("\nKemungkinan penyebab:")
        print("1. Internet tidak stabil")
        print("2. KaggleHub gagal download")
        print("3. Dataset Kaggle sedang bermasalah")

        print("\nSolusi alternatif:")
        print(
            "Download manual dari Kaggle lalu extract ke folder dataset/"
        )


if __name__ == "__main__":
    download_dataset()