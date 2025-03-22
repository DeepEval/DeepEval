import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Load CIFAR-10 dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Define the architecture
    model = models.Sequential([
        # Main path
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Branch path
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Combine paths
        layers.Add(),

        # Flatten before fully connected layers
        layers.Flatten(),

        # Fully connected paths
        layers.Dense(512, activation='relu'),
        layers.Dense(10)
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model

# Build the model
model = dl_model()