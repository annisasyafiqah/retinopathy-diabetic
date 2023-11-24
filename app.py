import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from PIL import Image, ImageTk, ImageFilter
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import numpy as np

# Load your trained model here
model = tf.keras.models.load_model('modelDR.h5')
class_names = ['Diabetic', 'Normal']

def open_image():
    # Buka dialog pemilihan file gambar
    file_path = filedialog.askopenfilename()

    if file_path:
        # Perform image processing and prediction
        img_process, prediction = process_image(file_path)
        
        # Display the processed image and prediction
        display_image(img_process)
        display_prediction(prediction)

def process_image(image_path):
    IMG_SIZE = 512
    Pred_result_Path = "Prediksi.png"

    img = Image.open(image_path)
    img_process = img.convert('L')
    img_process = img_process.resize((IMG_SIZE, IMG_SIZE))
    img_process = Image.blend(img_process, img_process.filter(ImageFilter.GaussianBlur(40)), alpha=0.5)
    img_process.save(Pred_result_Path)

    img_pred = load_img(Pred_result_Path, target_size=(150, 150))
    img_array = img_to_array(img_pred)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.sigmoid(predictions[0])

    return img_process, score

def display_image(image):
    photo = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0, image=photo, anchor=tk.NW)
    canvas.image = photo

def display_prediction(prediction):
    prediction_label.config(text="Prediction: {}".format(class_names[np.argmax(prediction)]))


# Inisialisasi window Tkinter
window = tk.Tk()
window.title("Deteksi Diabetik Retinopati")
window.geometry("800x600")

frame_width = 600
frame_height = 600

# Buat tombol untuk membuka gambar
btn_open = tk.Button(window, text="Pilih Citra Retina", command=open_image, font=("Arial", 14), bg="gray", fg="white")
btn_open.pack(pady=20)

prediction_label = tk.Label(window, text="Prediction: ", font=("Arial", 16), fg="black")  # Definition for prediction label
prediction_label.pack()

# confidence_label = tk.Label(window, text="Confidence: ", font=("Arial", 14), fg="black")  # Definition for confidence label
# confidence_label.pack()

# Buat canvas untuk menampilkan gambar
canvas = tk.Canvas(window, width=frame_width, height=frame_height, bg="white" )
canvas.pack()

# Jalankan main loop aplikasi Tkinter
window.mainloop()