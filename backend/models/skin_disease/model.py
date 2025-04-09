import tensorflow as tf
import numpy as np
import cv2

MODEL_PATH = "backend/models/skin_disease/saved_model/model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (299, 299))  # Xception input size
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_skin_disease(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return predicted_class
