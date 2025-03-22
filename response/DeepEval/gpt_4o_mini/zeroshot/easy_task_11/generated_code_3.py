import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer for 28x28 images with a single color channel
    input_layer = layers.Input(shape=(28, 28, 1))

    # Average Pooling Layer with 5x5 window and 3x3 stride
    x = layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(input_layer)

    # 1x1 Convolutional Layer
    x = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(x)

    # Flatten the feature maps
    x = layers.Flatten()(x)

    # First Fully Connected Layer
    x = layers.Dense(128, activation='relu')(x)

    # Dropout Layer to reduce overfitting
    x = layers.Dropout(0.5)(x)

    # Second Fully Connected Layer
    x = layers.Dense(64, activation='relu')(x)

    # Output Layer with 10 classes for classification
    output_layer = layers.Dense(10, activation='softmax')(x)

    # Construct the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of how to create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Print the model summary
model.summary()