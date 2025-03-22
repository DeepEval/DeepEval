import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the input shape
input_shape = (32, 32, 3)

# Define the model
model = keras.Sequential([
    layers.Lambda(lambda x: tf.split(x, 3, axis=-1)),
    layers.Lambda(lambda x: x[0] + x[1]),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.Conv2D(32, (1, 1), activation='relu'),
    layers.Lambda(lambda x: x[0] + x[1]),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Conv2D(64, (1, 1), activation='relu'),
    layers.Lambda(lambda x: x[0] + x[1]),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Conv2D(128, (1, 1), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))