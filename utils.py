import os
import json

dataset_path = './dataset2/train'
class_names = sorted(os.listdir(dataset_path))
class_indices = {class_name: i for i, class_name in enumerate(class_names)}

with open('class_labels.json', 'w') as f:
    json.dump(class_indices, f)


'''Python file for predicting a single image class'''
# import numpy as np
# import json
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.models import load_model

# def load_and_preprocess_image(image_path, img_width, img_height):
#     img = image.load_img(image_path, target_size=(img_width, img_height))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0
#     return img_array

# # Load the class labels
# with open('class_labels.json', 'r') as f:
#     class_labels = json.load(f)

# # Set the path to your ASL image
# image_path = './A_ownhand.jpg'

# # Load the trained model
# model = load_model('asl_cnn_model.h5')

# # Load and preprocess the image
# img_width, img_height = 64, 64  # Change the dimensions to match the training input shape
# img_array = load_and_preprocess_image(image_path, img_width, img_height)

# # Make a prediction
# prediction = model.predict(img_array)
# predicted_class_index = np.argmax(prediction)

# # Get the class label from the index
# predicted_class_label = list(class_labels.keys())[list(class_labels.values()).index(predicted_class_index)]

# print(f"Predicted class index: {predicted_class_index}")
# print(f"Predicted class label: {predicted_class_label}")



'''GUI without Feedback Learning'''
# import os
# import cv2
# import numpy as np
# from tkinter import Tk, Button, Label, filedialog, messagebox
# from tkinter.ttk import Frame, Label, Button, Style
# from ttkthemes import ThemedTk
# from keras.models import load_model
# from keras.preprocessing import image

# # Load the saved model
# model = load_model('asl_cnn_model.h5')

# # Preprocess image
# def preprocess_image(image_path=None, img=None):
#     if image_path:
#         img = cv2.imread(image_path)
#     img = cv2.resize(img, (64, 64))
#     img = img.astype('float32')
#     img /= 255
#     img = np.expand_dims(img, axis=0)
#     return img

# # Classify image
# def classify_image(img):
#     predictions = model.predict(img)
#     class_index = np.argmax(predictions, axis=1)[0]
#     confidence = predictions[0][class_index]
#     return class_index, confidence

# # Function to handle button click
# def on_button_click():
#     file_path = filedialog.askopenfilename()
#     if not file_path:
#         return
#     img = preprocess_image(file_path)
#     class_index, confidence = classify_image(img)

#     # Replace this line with the list of your ASL alphabet labels
#     class_labels = ['A', 'B', 'C']  # You should update this list with all the ASL alphabet labels that your model can predict

#     messagebox.showinfo("Result", f"Predicted Class: {class_labels[class_index]}\nConfidence: {confidence * 100:.2f}%")

# def capture_image():
#     cap = cv2.VideoCapture(0)

#     while True:
#         ret, frame = cap.read()
#         cv2.imshow('Capture Image', frame)
#         key = cv2.waitKey(1)
#         if key == ord('c'):  # Press 'c' to capture the image
#             img = frame.copy()
#             break
#         elif key == ord('q'):  # Press 'q' to quit without capturing
#             cap.release()
#             cv2.destroyAllWindows()
#             return

#     cap.release()
#     cv2.destroyAllWindows()

#     img = preprocess_image(img=img)
#     class_index, confidence = classify_image(img)

#     # Replace this line with the list of your ASL alphabet labels
#     class_labels = ['A', 'B', 'C']  # You should update this list with all the ASL alphabet labels that your model can predict

#     messagebox.showinfo("Result", f"Predicted Class: {class_labels[class_index]}\nConfidence: {confidence * 100:.2f}%")

# # Create GUI
# root = ThemedTk(theme="equilux")  # You can choose from various themes available in ttkthemes
# root.title("ASL Interpreter")

# # Set window size and position
# window_width, window_height = 400, 200
# screen_width, screen_height = root.winfo_screenwidth(), root.winfo_screenheight()
# position_x, position_y = (screen_width // 2) - (window_width // 2), (screen_height // 2) - (window_height // 2)
# root.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")

# main_frame = Frame(root)
# main_frame.pack(padx=10, pady=10)

# label = Label(main_frame, text="ASL Interpreter", font=("Arial", 20, "bold"))
# label.pack(pady=20)

# button_choose = Button(main_frame, text="Choose Image", style="TButton", command=on_button_click)
# button_choose.pack(pady=10)

# button_capture = Button(main_frame, text="Capture Image", style="TButton", command=capture_image)
# button_capture.pack(pady=10)

# root.mainloop()
