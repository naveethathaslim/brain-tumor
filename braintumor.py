# braintumor.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load trained model
model = load_model("model.h5")
print( model.input_shape)

# Class labels (update based on your dataset)
CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

def predict_tumor(img_path):
    img = image.load_img(img_path, target_size=(150, 150,3))
    img_array = image.img_to_array(img) / 255.0

    # Flatten to match model input (34848,)
    img_array = img_array.flatten()
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100

    return predicted_class, confidence
