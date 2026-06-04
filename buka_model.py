import tensorflow as tf

# 1. Load model .h5
model_path = "models/food_model.h5"
model = tf.keras.models.load_model(model_path)

# 2. Tampilkan ringkasan arsitektur model (Daftar layer & jumlah parameter)
print("=== RINGKASAN MODEL ===")
model.summary()

# 3. Melihat detail konfigurasi layer dalam format teks/JSON
print("\n=== DETAIL KONFIGURASI ===")
for i, layer in enumerate(model.layers):
    print(f"Layer {i}: Name={layer.name}, Type={type(layer).__name__}, Trainable={layer.trainable}")

# 4. Mengakses bobot (weights) dari layer tertentu (misal layer Dense pertama)
# dense_layer = model.get_layer("dense")
# weights, biases = dense_layer.get_weights()
# print(f"Shape Bobot: {weights.shape}, Shape Bias: {biases.shape}")
