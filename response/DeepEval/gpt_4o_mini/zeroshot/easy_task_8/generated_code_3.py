import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer for 28x28 grayscale images
    input_layer = layers.Input(shape=(28, 28, 1))

    # Depthwise separable convolution
    x = layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    x = layers.Dropout(0.2)(x)  # Dropout layer to mitigate overfitting

    # 1x1 Convolution
    x = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(x)
    x = layers.Dropout(0.2)(x)  # Dropout layer to mitigate overfitting

    # Flatten the output
    x = layers.Flatten()(x)

    # Fully connected layer
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)  # Dropout layer to mitigate overfitting

    # Output layer with softmax activation for classification
    output_layer = layers.Dense(10, activation='softmax')(x)

    # Construct the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255.0
x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.0
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Create the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Fit the model (optional, can be executed if needed)
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1)