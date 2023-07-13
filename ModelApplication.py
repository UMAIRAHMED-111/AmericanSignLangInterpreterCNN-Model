import os
import cv2
import numpy as np
from tkinter import Tk, Toplevel, Button, Entry, Label, StringVar, filedialog, messagebox
from tkinter.ttk import Frame, Label, Button, Style
from ttkthemes import ThemedTk
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image, ImageTk, ImageDraw, ImageFont
from PIL.Image import Resampling  # Add this import

# Load the saved model
model = load_model('asl_cnn_model.h5')

# Preprocess image
def preprocess_image(image_path=None, img=None):
    if image_path:
        img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    img = img.astype('float32')
    img /= 255
    img = np.expand_dims(img, axis=0)
    return img

# Classify image
def classify_image(img):
    predictions = model.predict(img)
    class_index = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][class_index]
    return class_index, confidence

def display_image(img, class_index, confidence, window_title):
    class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    img_pil = img_pil.resize((256, 256), Resampling.LANCZOS)

    draw = ImageDraw.Draw(img_pil)  # Create a separate ImageDraw object
    font = ImageFont.truetype("arial.ttf", 20)
    draw.text((10, 10), f"Predicted: {class_labels[class_index]}", font=font, fill=(255, 255, 255))
    draw.text((10, 40), f"Confidence: {confidence * 100:.2f}%", font=font, fill=(255, 255, 255))

    img_tk = ImageTk.PhotoImage(img_pil)  # Use the original img_pil object here

    image_window = Toplevel(root)
    image_window.title(window_title)
    label = Label(image_window, image=img_tk)
    label.image = img_tk
    label.pack()


def capture_image():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        cv2.imshow('Capture Image', frame)
        key = cv2.waitKey(1)
        if key == ord('c'):  # Press 'c' to capture the image
            img = frame.copy()
            break
        elif key == ord('q'):  # Press 'q' to quit without capturing
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    img_preprocessed = preprocess_image(img=img)
    class_index, confidence = classify_image(img_preprocessed)
    display_image(img, class_index, confidence, "Captured Image Prediction")

def on_button_click():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    img = cv2.imread(file_path)
    img_preprocessed = preprocess_image(img=img)
    class_index, confidence = classify_image(img_preprocessed)
    display_image(img, class_index, confidence, "Selected Image Prediction")

# Create GUI
root = ThemedTk(theme="equilux")
root.title("ASL Interpreter")

# Set window size and position
window_width, window_height = 400, 200
screen_width, screen_height = root.winfo_screenwidth(), root.winfo_screenheight()
position_x, position_y = (screen_width // 2) - (window_width // 2), (screen_height // 2) - (window_height // 2)
root.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")

main_frame = Frame(root)
main_frame.pack(padx=10, pady=10)

label = Label(main_frame, text="ASL Interpreter", font=("Arial", 20, "bold"))
label.pack(pady=10, padx=80)

button_choose = Button(main_frame, text="Choose Image", style="TButton", command=on_button_click)
button_choose.pack(pady=10)

button_capture = Button(main_frame, text="Capture Image", style="TButton", command=capture_image)
button_capture.pack(pady=10)

root.mainloop()