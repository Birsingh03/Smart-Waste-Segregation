import tensorflow as tf
import matplotlib.pyplot as plt

# Load dataset
dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/standardized_256",
    image_size=(224, 224),
    batch_size=9,
    shuffle=True
)

class_names = dataset.class_names

# Display 9 images
for images, labels in dataset.take(1):

    plt.figure(figsize=(10,10))

    for i in range(9):
        ax = plt.subplot(3,3,i+1)

        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

plt.show()