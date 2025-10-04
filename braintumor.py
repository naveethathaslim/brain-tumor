import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load trained model - update path if different
model = load_model("model.h5")

# Class names (update according to your dataset)
CLASS_NAMES = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")         # Convert to RGB
    img = img.resize((69, 69))                           # Resize to model input size
    img_array = np.array(img) / 255.0                    # Normalize pixels to [0,1]
    img_array = np.expand_dims(img_array, axis=0)        # Add batch dimension
    return img_array

def predict_tumor(image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)               # Model prediction
    pred_class = np.argmax(predictions, axis=1)[0]       # Get class index
    confidence = float(np.max(predictions) * 100)        # Get confidence %
    return CLASS_NAMES[pred_class], confidence
