import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer for the MNIST images (28x28 pixels, 1 channel)
    input_shape = (28, 28, 1)
    inputs = layers.Input(shape=input_shape)

    # First convolutional block
    x1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x1 = layers.MaxPooling2D((2, 2))(x1)

    # Second convolutional block
    x2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x2 = layers.MaxPooling2D((2, 2))(x2)

    # Ensure the output from both blocks has the same shape
    x1_shape = layers.Conv2D(32, (1, 1), activation='linear', padding='same')(x1)
    x2_shape = layers.Conv2D(32, (1, 1), activation='linear', padding='same')(x2)

    # Combine the outputs of both blocks
    combined = layers.Add()([x1_shape, x2_shape])

    # Flatten the combined output
    x = layers.Flatten()(combined)

    # Fully connected layer for classification
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example of how to compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255.0
x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Train the model (example, can be adjusted)
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))