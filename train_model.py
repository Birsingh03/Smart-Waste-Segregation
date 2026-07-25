import tensorflow as tf
import os

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential

# Load the dataset
dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/standardized_256",
    image_size=(224, 224),
    batch_size=32,
    shuffle=True
)

# Get class names
class_names = dataset.class_names

print("Classes:")
print(class_names)

# Split dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/standardized_256",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(224, 224),
    batch_size=32
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/standardized_256",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(224, 224),
    batch_size=32
)

class_names = train_dataset.class_names

# Load pre-trained ResNet50
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze ResNet50
base_model.trainable = False

# Build model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation="relu"),
    Dropout(0.3),
    Dense(len(class_names), activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Print summary
model.summary()

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=5
)

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

# Save the model

os.makedirs("models", exist_ok=True)

model.save("models/garbage_classifier.keras")

print("Model saved successfully!")