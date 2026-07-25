import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("models/garbage_classifier.keras")

# Class names (must be in the same order as training)
class_names = [
    "battery",
    "biological",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash"
]

# Map each class to a bin
BIN_MAPPING = {
    "battery": "General Waste Bin",
    "biological": "Organic Bin",
    "cardboard": "Dry Waste Bin",
    "clothes": "General Waste Bin",
    "glass": "Dry Waste Bin",
    "metal": "Dry Waste Bin",
    "paper": "Dry Waste Bin",
    "plastic": "Plastic Bin",
    "shoes": "General Waste Bin",
    "trash": "General Waste Bin"
}

# Image path
image_path = input("Enter image path: ")

# Load image
img = tf.keras.utils.load_img(
    image_path,
    target_size=(224,224)
)

img_array = tf.keras.utils.img_to_array(img)

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)

predicted_class = class_names[np.argmax(prediction)]

confidence = np.max(prediction) * 100

print("\nPrediction:", predicted_class)
print(f"Confidence: {confidence:.2f}%")
print("Open:", BIN_MAPPING[predicted_class])