import tensorflow as tf
import numpy as np
import cv2

def generate_gradcam(model, img_path, target_size=(128, 128), last_conv_layer_name="Conv_1", pred_index=None):
    """
    Membangun peta aktivasi kelas (Grad-CAM) untuk memvisualisasikan daerah gambar
    yang paling berkontribusi terhadap hasil klasifikasi CNN.

    Parameter:
    - model: Model Keras Sequential penuh
    - img_path: Path ke file gambar input
    - target_size: Ukuran input model (default 128x128)
    - last_conv_layer_name: Nama layer konvolusi terakhir di base model (default Conv_1)
    - pred_index: Indeks kelas target (jika None, ambil kelas dengan probabilitas tertinggi)

    Return:
    - overlay: Gambar BGR hasil overlay heatmap Grad-CAM pada gambar asli
    """
    # 1. Baca gambar asli
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Grad-CAM: Gagal membaca gambar dari {img_path}")
    
    h_orig, w_orig = img_bgr.shape[:2]
    
    # 2. Preprocessing gambar agar sesuai input model
    from predict import _clahe_enhance
    enhanced = _clahe_enhance(img_bgr)
    resized = cv2.resize(enhanced, target_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    img_array = np.expand_dims(rgb, axis=0)
    
    # 3. Pisahkan base model (MobileNetV2) dari sequential head
    base_model = model.layers[0]
    
    # Dapatkan layer konvolusi terakhir
    try:
        conv_layer = base_model.get_layer(last_conv_layer_name)
    except ValueError:
        # Fallback ke out_relu jika Conv_1 tidak ditemukan
        last_conv_layer_name = "out_relu"
        conv_layer = base_model.get_layer(last_conv_layer_name)

    # Buat sub-model untuk merekam output layer konvolusi & output base model
    base_grad_model = tf.keras.models.Model(
        [base_model.inputs], [conv_layer.output, base_model.output]
    )
    
    # 4. Rekam gradient menggunakan GradientTape
    with tf.GradientTape() as tape:
        base_conv_outputs, base_preds = base_grad_model(img_array)
        # Teruskan output base model ke sequential head layers
        x = base_preds
        for layer in model.layers[1:]:
            x = layer(x)
        preds = x
        
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Hitung gradien neuron kelas target terhadap output layer konvolusi
    grads = tape.gradient(class_channel, base_conv_outputs)
    
    # Rata-ratakan gradien pada setiap channel (Global Average Pooling pada gradien)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Kalikan setiap channel dengan bobot kepentingannya, lalu jumlahkan (Weighted Sum)
    base_conv_outputs = base_conv_outputs[0]
    heatmap = base_conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU pada heatmap (hanya ambil fitur yang berdampak positif terhadap kelas target)
    heatmap = tf.maximum(heatmap, 0)
    
    # Normalisasi heatmap ke rentang [0, 1]
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    heatmap = heatmap.numpy()

    # 5. Overlay heatmap pada gambar asli
    # Resize heatmap ke ukuran gambar asli menggunakan interpolasi bilinear
    heatmap_resized = cv2.resize(heatmap, (w_orig, h_orig))
    heatmap_u8 = np.uint8(255 * heatmap_resized)
    
    # Beri warna heatmap dengan skema warna JET (biru = dingin/tidak penting, merah = panas/penting)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    
    # Blend gambar asli dengan heatmap (60% gambar asli, 40% heatmap)
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap_color, 0.4, 0)
    
    return overlay
