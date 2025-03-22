import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 pixels with 3 color channels (RGB)
    
    # Input Layer
    inputs = layers.Input(shape=input_shape)

    # First Block - Feature Extraction
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    
    # Residual Connection
    x = layers.Add()([x, inputs])
    
    # Second Block - Channel Compression
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(32, activation='relu')(x)  # First Fully Connected Layer
    x = layers.Dense(32, activation='relu')(x)  # Second Fully Connected Layer
    
    # Reshape to match the feature map
    x = layers.Reshape((1, 1, 32))(x)  # Reshape to 1x1x32 to match dimensions for multiplication
    x = layers.Multiply()([x, inputs])  # Multiply with the original input

    # Flatten and Output Layer
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)  # CIFAR-10 has 10 classes

    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
if __name__ == "__main__":
    model = dl_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Train the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=64)