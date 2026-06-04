import kagglehub
import shutil
import os

print("START DOWNLOAD...")

path = kagglehub.dataset_download(
    "anasfikrihanif/indonesian-food-and-drink-nutrition-dataset"
)

print("DOWNLOAD SELESAI")
print("PATH:", path)

target = "dataset/indonesian_food"

os.makedirs(target, exist_ok=True)

for file in os.listdir(path):

    if file.endswith(".csv"):

        shutil.copy(
            os.path.join(path, file),
            os.path.join(target, file)
        )

        print("COPY:", file)

print("SELESAI")