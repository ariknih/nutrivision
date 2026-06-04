import os
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


TRAIN_PATH = "dataset/dataset_gambar/train"
VALID_PATH = "dataset/dataset_gambar/valid"

MODEL_PATH = "models/food_model.h5"
BEST_MODEL_PATH = "models/food_model_best.h5"


def train_model():
    os.makedirs("models", exist_ok=True)

    # Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        fill_mode='nearest'
    )

    valid_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_data = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical'
    )

    valid_data = valid_datagen.flow_from_directory(
        VALID_PATH,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical'
    )

    print("\nCLASS:")
    print(train_data.class_indices)

    # MobileNetV2 base (Transfer Learning)
    base_model = MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights='imagenet'
    )

    # Phase 1: freeze seluruh base model
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(train_data.num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Callbacks
    callbacks = [
        # Simpan model terbaik berdasarkan val_accuracy
        ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Kurangi learning rate jika val_loss stagnan 3 epoch
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        # Stop lebih awal jika val_accuracy tidak naik 7 epoch berturut-turut
        EarlyStopping(
            monitor='val_accuracy',
            patience=7,
            restore_best_weights=True,
            verbose=1
        )
    ]

    print("\n==== PHASE 1: Training head layers (20 epoch) ====")
    history = model.fit(
        train_data,
        validation_data=valid_data,
        epochs=20,
        callbacks=callbacks
    )

    # Phase 2: Fine-tuning — unfreeze 30 layer terakhir base model
    print("\n==== PHASE 2: Fine-tuning top layers MobileNetV2 ====")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Learning rate lebih kecil untuk fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks_ft = [
        ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=6,
            restore_best_weights=True,
            verbose=1
        )
    ]

    model.fit(
        train_data,
        validation_data=valid_data,
        epochs=10,   # fine-tuning 10 epoch tambahan
        callbacks=callbacks_ft
    )

    # Simpan model final (dengan bobot terbaik sudah di-restore oleh EarlyStopping)
    model.save(MODEL_PATH)
    print(f"\nModel final disimpan ke: {MODEL_PATH}")
    print(f"Model terbaik (val_accuracy) disimpan ke: {BEST_MODEL_PATH}")

    with open("models/class_indices.txt", "w") as f:
        f.write(str(train_data.class_indices))

    print("\nMODEL BERHASIL DISIMPAN ✅")


if __name__ == "__main__":
    train_model()