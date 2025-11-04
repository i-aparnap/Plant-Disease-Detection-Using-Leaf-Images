import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Load model and class names
model = tf.keras.models.load_model("mobilenetv2_best.h5")
class_indices = np.load("class_indices.npy", allow_pickle=True).item()
class_names = list(class_indices.keys())

# --- Main window setup ---
root = tk.Tk()
root.title("🌿 Plant Disease Detection")
root.geometry("800x600")

# Background image
bg_image = Image.open("background.jpg")
bg_photo = ImageTk.PhotoImage(bg_image.resize((800, 600)))
bg_label = tk.Label(root, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# Title
title_label = tk.Label(root, text="🌱 Plant Disease Detector",
                       font=("Helvetica", 24, "bold"),
                       bg="#ffffff", fg="#2e7d32")
title_label.pack(pady=50)

# --- Result frame (hidden initially) ---
result_frame = tk.Frame(root, bg="#ffffff")
result_frame.pack(pady=10)
result_frame.pack_forget()  # hide at start

# Thumbnail
icon_label = tk.Label(result_frame, bg="#ffffff")
icon_label.pack(side="left", padx=15)

# Result text
result_label = tk.Label(result_frame, text="",
                        font=("Helvetica", 16),
                        bg="#ffffff", fg="#1b5e20", justify="left")
result_label.pack(side="left")

# --- Upload and predict function ---
def upload_and_predict():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        # Prepare image for model
        img = Image.open(file_path)
        resized_img = img.resize((224, 224))
        img_array = np.array(resized_img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Temporary message
        result_label.config(text="Detecting disease... please wait 🌿", fg="#388e3c")
        result_frame.pack(pady=10)
        root.update()

        # Model prediction
        prediction = model.predict(img_array)
        pred_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Display thumbnail (small version)
        thumb = img.resize((100, 100))
        thumb_photo = ImageTk.PhotoImage(thumb)
        icon_label.config(image=thumb_photo)
        icon_label.image = thumb_photo

        # Display result text
        result_label.config(
            text=f"Prediction: {pred_class}\nConfidence: {confidence:.2f}%",
            fg="#2e7d32"
        )

        # Reveal result frame (if hidden)
        result_frame.pack(pady=10)

# --- Upload button ---
upload_btn = tk.Button(root, text="Upload Leaf Image",
                       command=upload_and_predict,
                       font=("Helvetica", 14),
                       bg="#4caf50", fg="white",
                       padx=20, pady=10)
upload_btn.pack(pady=30)

root.mainloop()