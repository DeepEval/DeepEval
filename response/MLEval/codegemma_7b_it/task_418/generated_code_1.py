# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the model architecture
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Input layer, flatten 28x28 images
    layers.Dense(128, activation='relu'),  # Hidden layer with 128 units and ReLU activation
    layers.Dense(10, activation='softmax')  # Output layer with 10 units (for 10 classes)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define the training data
(train_images, train_labels), (_, _) = keras.datasets.mnist.load_data()
train_images = train_images / 255.0  # Normalize pixel values

# Train the model
history = model.fit(train_images, train_labels, epochs=10)

# Print the final accuracy
print("Final accuracy:", history.history['accuracy'][-1])

# Return the output (optional)
output = history.history['accuracy'][-1]
return output