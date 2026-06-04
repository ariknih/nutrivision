import os
import threading

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np

from preprocessing import spatial_preprocessing, frequency_preprocessing
from segmentation import segment_food
from feature_extraction import extract_features
from calorie import estimate_weight, get_nutrition_data
from gradcam import generate_gradcam
from predict import _load_model


class NutriVisionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("NutriVision AI")
        self.root.geometry("1366x780")
        self.root.configure(bg="#0f172a")
        self.root.resizable(True, True)
        self.root.minsize(1200, 720)

        self.image_path = None
        self.input_img_tk = None
        self.output_img_tk = None
        self.gradcam_img_tk = None
        self.fft_img_tk = None
        self._processing = False

        self.build_ui()

    # ──────────────── HELPERS ──────────────────────────────────────────────
    def bind_hover(self, button, hover_bg, normal_bg):
        button.bind("<Enter>", lambda e: button.configure(bg=hover_bg))
        button.bind("<Leave>", lambda e: button.configure(bg=normal_bg))

    def show_image(self, cv_img, panel, size):
        if len(cv_img.shape) == 2:
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img = pil_img.resize(size, Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(pil_img)
        panel.configure(image=img_tk, text="", width=size[0], height=size[1])
        panel.image = img_tk
        return img_tk

    # ──────────────── BUILD UI ─────────────────────────────────────────────
    def build_ui(self):
        # ── Header ──────────────────────────────────────────────────────────
        header = tk.Frame(self.root, bg="#111827", height=80)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="🍽  NutriVision AI",
            font=("Segoe UI", 26, "bold"),
            fg="#f8fafc",
            bg="#111827"
        ).pack(side="left", padx=28, pady=15)

        tk.Label(
            header,
            text="PCD: Gaussian Lowpass Filter (GLPF)  ·  Explainable AI (Grad-CAM)  ·  CNN + TTA  ·  Segmentasi & Kontur",
            font=("Segoe UI", 11, "italic"),
            fg="#38bdf8",
            bg="#111827"
        ).pack(side="left", padx=10, pady=20)

        # ── Status bar (top-right) ───────────────────────────────────────────
        self.status_var = tk.StringVar(value="Siap")
        tk.Label(
            header,
            textvariable=self.status_var,
            font=("Segoe UI", 10),
            fg="#64748b",
            bg="#111827"
        ).pack(side="right", padx=28)

        # ── Button bar ──────────────────────────────────────────────────────
        button_bar = tk.Frame(self.root, bg="#0f172a", height=72)
        button_bar.pack(fill="x")
        button_bar.pack_propagate(False)

        # Centre the buttons
        btn_frame = tk.Frame(button_bar, bg="#0f172a")
        btn_frame.pack(expand=True)

        self.btn_upload = tk.Button(
            btn_frame,
            text="📁  UPLOAD GAMBAR",
            command=self.upload_image,
            bg="#2563eb",
            fg="white",
            font=("Segoe UI", 12, "bold"),
            width=20,
            height=2,
            bd=0,
            cursor="hand2",
            activebackground="#1d4ed8",
            activeforeground="white"
        )
        self.btn_upload.pack(side="left", padx=10, pady=12)
        self.bind_hover(self.btn_upload, "#1d4ed8", "#2563eb")

        self.btn_detect = tk.Button(
            btn_frame,
            text="🔍  DETEKSI & ANALISIS",
            command=self.start_process,
            bg="#16a34a",
            fg="white",
            font=("Segoe UI", 12, "bold"),
            width=22,
            height=2,
            bd=0,
            cursor="hand2",
            activebackground="#15803d",
            activeforeground="white"
        )
        self.btn_detect.pack(side="left", padx=10, pady=12)
        self.bind_hover(self.btn_detect, "#15803d", "#16a34a")

        # (Evaluasi Model button removed)

        self.btn_reset = tk.Button(
            btn_frame,
            text="🔄  RESET",
            command=self.reset,
            bg="#475569",
            fg="white",
            font=("Segoe UI", 12, "bold"),
            width=10,
            height=2,
            bd=0,
            cursor="hand2",
            activebackground="#334155",
            activeforeground="white"
        )
        self.btn_reset.pack(side="left", padx=10, pady=12)
        self.bind_hover(self.btn_reset, "#334155", "#475569")

        # ── Body ─────────────────────────────────────────────────────────────
        body = tk.Frame(self.root, bg="#0f172a")
        body.pack(fill="both", expand=True, padx=16, pady=(0, 10))
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.columnconfigure(2, weight=1)
        body.rowconfigure(0, weight=1)

        # ─ Column 0: Left panel (Input & FFT) ──────────────────────────────────
        left = tk.Frame(body, bg="#1e293b",
                        highlightbackground="#334155", highlightthickness=1)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        # Title Input
        tk.Label(
            left,
            text="📸 Gambar Input",
            font=("Segoe UI", 13, "bold"),
            fg="#f1f5f9",
            bg="#1e293b"
        ).pack(pady=(10, 2))

        self.input_panel = tk.Label(
            left,
            text="Belum ada gambar\nKlik  📁 UPLOAD GAMBAR",
            font=("Segoe UI", 10),
            fg="#64748b",
            bg="#020617",
            highlightbackground="#334155",
            highlightthickness=1,
            wraplength=340
        )
        self.input_panel.pack(padx=12, pady=4, fill="both", expand=True)

        # Title Gaussian
        tk.Label(
            left,
            text="🟢 Gaussian Lowpass Filter (GLPF)",
            font=("Segoe UI", 13, "bold"),
            fg="#f1f5f9",
            bg="#1e293b"
        ).pack(pady=(10, 2))

        self.fft_panel = tk.Label(
            left,
            text="Hasil Filter Gaussian Lowpass\n(tampil setelah upload gambar)",
            font=("Segoe UI", 10),
            fg="#64748b",
            bg="#020617",
            highlightbackground="#334155",
            highlightthickness=1,
            wraplength=340
        )
        self.fft_panel.pack(padx=12, pady=(4, 12), fill="both", expand=True)

        # ─ Column 1: Middle panel (Segmentation & Grad-CAM) ──────────────────
        middle = tk.Frame(body, bg="#1e293b",
                          highlightbackground="#334155", highlightthickness=1)
        middle.grid(row=0, column=1, sticky="nsew", padx=6)

        # Title Contour
        tk.Label(
            middle,
            text="🟢 Segmentasi & Kontur Objek",
            font=("Segoe UI", 13, "bold"),
            fg="#f1f5f9",
            bg="#1e293b"
        ).pack(pady=(10, 2))

        self.output_panel = tk.Label(
            middle,
            text="Hasil deteksi kontur\n(tampil setelah analisis)",
            font=("Segoe UI", 10),
            fg="#64748b",
            bg="#020617",
            highlightbackground="#334155",
            highlightthickness=1,
            wraplength=340
        )
        self.output_panel.pack(padx=12, pady=4, fill="both", expand=True)

        # Title Grad-CAM
        tk.Label(
            middle,
            text="🔥 Visualisasi Grad-CAM (CNN Focus)",
            font=("Segoe UI", 13, "bold"),
            fg="#f1f5f9",
            bg="#1e293b"
        ).pack(pady=(10, 2))

        self.gradcam_panel = tk.Label(
            middle,
            text="Heatmap Peta Aktivasi Kelas\n(tampil setelah analisis)",
            font=("Segoe UI", 10),
            fg="#64748b",
            bg="#020617",
            highlightbackground="#334155",
            highlightthickness=1,
            wraplength=340
        )
        self.gradcam_panel.pack(padx=12, pady=(4, 12), fill="both", expand=True)

        # ─ Column 2: Right panel (Results Text) ──────────────────────────────
        right = tk.Frame(body, bg="#1e293b",
                         highlightbackground="#334155", highlightthickness=1)
        right.grid(row=0, column=2, sticky="nsew", padx=(6, 0))

        tk.Label(
            right,
            text="📋 Laporan Analisis & Nutrisi",
            font=("Segoe UI", 13, "bold"),
            fg="#f1f5f9",
            bg="#1e293b"
        ).pack(pady=(10, 4))

        # Result text box
        text_frame = tk.Frame(right, bg="#1e293b")
        text_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        scrollbar = tk.Scrollbar(text_frame, bg="#1e293b", troughcolor="#0f172a",
                                 activebackground="#475569")
        scrollbar.pack(side="right", fill="y")

        self.result_text = tk.Text(
            text_frame,
            bg="#020617",
            fg="#e2e8f0",
            font=("Consolas", 10),
            bd=0,
            padx=12,
            pady=10,
            highlightbackground="#334155",
            highlightthickness=1,
            insertbackground="white",
            wrap="word",
            yscrollcommand=scrollbar.set
        )
        self.result_text.pack(fill="both", expand=True)
        scrollbar.config(command=self.result_text.yview)

        # Configure text colour tags
        self.result_text.tag_config("header",
                                    foreground="#38bdf8",
                                    font=("Consolas", 10, "bold"))
        self.result_text.tag_config("food",
                                    foreground="#4ade80",
                                    font=("Consolas", 11, "bold"))
        self.result_text.tag_config("warn",
                                    foreground="#fbbf24",
                                    font=("Consolas", 10))
        self.result_text.tag_config("dim",
                                    foreground="#64748b"),
        self.result_text.tag_config("value",
                                    foreground="#e2e8f0")

        self._write_placeholder("Silakan upload gambar makanan terlebih dahulu.")

    # ──────────────── ACTIONS ──────────────────────────────────────────────
    def _write_placeholder(self, text):
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, text, "dim")
        self.result_text.configure(state="disabled")

    def reset(self):
        self.image_path = None
        self.input_img_tk = None
        self.output_img_tk = None
        self.gradcam_img_tk = None
        self.fft_img_tk = None
        self.input_panel.configure(
            image="",
            text="Belum ada gambar\nKlik  📁 UPLOAD GAMBAR",
            width=0, height=0
        )
        self.output_panel.configure(
            image="",
            text="Hasil deteksi kontur\n(tampil setelah analisis)",
            width=0, height=0
        )
        self.gradcam_panel.configure(
            image="",
            text="Heatmap Peta Aktivasi Kelas\n(tampil setelah analisis)",
            width=0, height=0
        )
        self.fft_panel.configure(
            image="",
            text="Hasil Filter Gaussian Lowpass\n(tampil setelah upload gambar)",
            width=0, height=0
        )
        self._write_placeholder("Silakan upload gambar makanan terlebih dahulu.")
        self.status_var.set("Siap")

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Pilih Gambar Makanan",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.webp")]
        )
        if not file_path:
            return

        self.image_path = file_path
        image = cv2.imread(file_path)
        if image is None:
            messagebox.showerror("Error", "Gambar tidak bisa dibaca.")
            return

        # Fit to panel while keeping aspect ratio
        panel_w, panel_h = 360, 240
        ih, iw = image.shape[:2]
        scale = min(panel_w / iw, panel_h / ih)
        disp_w, disp_h = int(iw * scale), int(ih * scale)

        # 1. Tampilkan Gambar Input
        self.input_img_tk = self.show_image(image, self.input_panel, (disp_w, disp_h))

        # 2. Tampilkan Gaussian Lowpass Filter secara instan!
        try:
            glpf_img = frequency_preprocessing(image)
            self.fft_img_tk = self.show_image(glpf_img, self.fft_panel, (disp_w, disp_h))
        except Exception as e:
            self.fft_panel.configure(text=f"Filter Gaussian gagal: {e}")

        # Reset output panels
        self.output_panel.configure(
            image="",
            text="Klik  🔍 DETEKSI & ANALISIS",
            width=0, height=18
        )
        self.gradcam_panel.configure(
            image="",
            text="Heatmap Peta Aktivasi Kelas\n(tampil setelah analisis)",
            width=0, height=18
        )
        self._write_placeholder("Gambar berhasil diupload.\nSekarang klik  🔍 DETEKSI & ANALISIS.")
        self.status_var.set(f"📂 {os.path.basename(file_path)}")

    # (Evaluation methods removed)

    # ──────────────── PROCESSING (runs in background thread) ─────────────
    def start_process(self):
        if self._processing:
            return
        if self.image_path is None:
            messagebox.showerror("Error", "Upload gambar dulu.")
            return

        self._processing = True
        self.btn_detect.configure(state="disabled", text="⏳  Memproses...")
        self.btn_upload.configure(state="disabled")
        self.status_var.set("⏳ Menganalisis gambar & membuat peta aktivasi Grad-CAM…")

        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, "⏳ Memuat model CNN & menjalankan TTA…\n\n", "dim")
        self.result_text.insert(tk.END,
            "Test-Time Augmentation sedang berjalan (8 variant).\n"
            "Ini membutuhkan beberapa detik…", "dim")
        self.result_text.configure(state="disabled")

        t = threading.Thread(target=self._process_worker, daemon=True)
        t.start()

    def _process_worker(self):
        try:
            image = cv2.imread(self.image_path)
            if image is None:
                raise ValueError("Gambar tidak bisa dibaca.")

            # ── CNN prediction with TTA & Color-profile correction ───────
            from predict import predict_top_k, _load_model
            
            top3 = predict_top_k(self.image_path, k=3)
            food_name_raw, confidence = top3[0]
            food_name_clean = food_name_raw.replace("_", " ").lower()

            # ── Spatial processing + segmentation ──────────────────────────
            spatial = spatial_preprocessing(image)
            segmented, mask, area = segment_food(spatial)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            boxed_image = spatial.copy()
            box_info = "Tidak terdeteksi"

            if len(contours) > 0:
                largest = max(contours, key=cv2.contourArea)
                x, y, bw, bh = cv2.boundingRect(largest)
                box_info = f"({x},{y}) → ({x+bw},{y+bh})"

                cv2.rectangle(
                    boxed_image, (x, y), (x + bw, y + bh),
                    (57, 255, 20), 3
                )
                label_text = food_name_clean
                cv2.putText(
                    boxed_image, label_text,
                    (x, max(y - 10, 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (57, 255, 20), 2, cv2.LINE_AA
                )
                # Confidence badge
                badge = f"{round(confidence * 100, 1)}%"
                cv2.putText(
                    boxed_image, badge,
                    (x, max(y - 30, 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 215, 0), 2, cv2.LINE_AA
                )

            # ── Feature extraction & nutrition ──────────────────────────────
            features = extract_features(segmented, mask)
            weight = estimate_weight(area)
            nutrition = get_nutrition_data(food_name_clean, weight)

            # ── Explainable AI (Grad-CAM) ──────────────────────────────────
            model, idx_to_class = _load_model()
            
            # Cari index dari kelas yang diprediksi
            pred_class_id = None
            for idx, name in idx_to_class.items():
                if name.replace("_", " ").lower() == food_name_clean:
                    pred_class_id = idx
                    break
            
            gradcam_overlay = generate_gradcam(model, self.image_path, pred_index=pred_class_id)

            # Schedule GUI update on the main thread
            self.root.after(0, self._update_ui,
                            boxed_image, top3, food_name_clean,
                            confidence, area, weight, nutrition,
                            features, box_info, gradcam_overlay)

        except Exception as exc:
            self.root.after(0, self._show_error, str(exc))

    def _update_ui(self, boxed_image, top3, food_name_clean, confidence,
                   area, weight, nutrition, features, box_info, gradcam_overlay):
        
        # Fit to panel size
        panel_w, panel_h = 360, 240
        
        # 1. Tampilkan Segmentasi & Kontur
        ih, iw = boxed_image.shape[:2]
        scale1 = min(panel_w / iw, panel_h / ih)
        self.output_img_tk = self.show_image(boxed_image, self.output_panel,
                                             (int(iw * scale1), int(ih * scale1)))

        # 2. Tampilkan Grad-CAM Heatmap overlay
        ih_g, iw_g = gradcam_overlay.shape[:2]
        scale2 = min(panel_w / iw_g, panel_h / ih_g)
        self.gradcam_img_tk = self.show_image(gradcam_overlay, self.gradcam_panel,
                                               (int(iw_g * scale2), int(ih_g * scale2)))

        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", tk.END)

        def write(text, tag="value"):
            self.result_text.insert(tk.END, text, tag)

        # ── CNN Results ─────────────────────────────────────────────────────
        write("══════ HASIL DETEKSI CNN (TTA) ══════\n", "header")
        write(f"Makanan     : ", "dim")
        write(f"{food_name_clean}\n", "food")
        write(f"Confidence  : ", "dim")

        conf_pct = round(confidence * 100, 2)
        conf_tag = "food" if conf_pct >= 70 else "warn"
        write(f"{conf_pct} %\n", conf_tag)

        if conf_pct < 60:
            write("⚠  Confidence rendah – hasil mungkin tidak akurat\n", "warn")

        write("\n── Top-3 Prediksi ──\n", "dim")
        for i, (lbl, sc) in enumerate(top3):
            pct = round(sc * 100, 1)
            bar_len = max(1, int(pct / 5))
            bar = "█" * bar_len
            tag = "food" if i == 0 else "dim"
            prefix = "►" if i == 0 else f" {i+1}"
            write(f"  {prefix} {lbl:<18} {pct:5.1f}%  {bar}\n", tag)

        # ── Segmentation ────────────────────────────────────────────────────
        write("\n══════ HASIL SEGMENTASI ══════\n", "header")
        write(f"Luas Objek  : ", "dim")
        write(f"{area:,} px²\n", "value")
        write(f"Bounding Box: ", "dim")
        write(f"{box_info}\n", "value")

        # ── Nutrition ───────────────────────────────────────────────────────
        write("\n══════ ANALISIS NUTRISI ══════\n", "header")
        write(f"Estimasi Berat : ", "dim")
        write(f"{weight} gram\n", "value")

        nutrients = [
            ("Kalori",      nutrition["calories"], "kkal"),
            ("Protein",     nutrition["protein"],  "g"),
            ("Lemak",       nutrition["fat"],      "g"),
            ("Karbohidrat", nutrition["carbohydrate"], "g"),
            ("Serat",       nutrition["fiber"],    "g"),
            ("Gula",        nutrition["sugar"],    "g"),
            ("Sodium",      nutrition["sodium"],   "mg"),
        ]
        for name, val, unit in nutrients:
            write(f"{name:<14}: ", "dim")
            write(f"{val} {unit}\n", "value")

        # ── Image features ──────────────────────────────────────────────────
        write("\n══════ FITUR CITRA ══════\n", "header")
        for key, val in features.items():
            write(f"{key:<16}: ", "dim")
            write(f"{val}\n", "value")

        self.result_text.configure(state="disabled")
        self.result_text.see("1.0")

        self.status_var.set(f"✅ Selesai — {food_name_clean} ({conf_pct}%)")
        self._reset_buttons()

    def _show_error(self, msg):
        messagebox.showerror("Error", msg)
        self.status_var.set("❌ Error")
        self._write_placeholder(f"Terjadi kesalahan:\n{msg}")
        self._reset_buttons()

    def _reset_buttons(self):
        self._processing = False
        self.btn_detect.configure(state="normal", text="🔍  DETEKSI & ANALISIS")
        self.btn_upload.configure(state="normal")


if __name__ == "__main__":
    root = tk.Tk()
    app = NutriVisionGUI(root)
    root.mainloop()