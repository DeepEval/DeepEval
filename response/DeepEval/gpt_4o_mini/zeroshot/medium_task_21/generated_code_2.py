import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # Branch 1: 1x1 Convolution
    branch1 = layers.Conv2D(32, (1, 1), activation='relu')(input_layer)
    branch1 = layers.Dropout(0.5)(branch1)

    # Branch 2: 1x1 Convolution followed by 3x3 Convolution
    branch2 = layers.Conv2D(32, (1, 1), activation='relu')(input_layer)
    branch2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(branch2)
    branch2 = layers.Dropout(0.5)(branch2)

    # Branch 3: 1x1 Convolution followed by two consecutive 3x3 Convolutions
    branch3 = layers.Conv2D(32, (1, 1), activation='relu')(input_layer)
    branch3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(branch3)
    branch3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(branch3)
    branch3 = layers.Dropout(0.5)(branch3)

    # Branch 4: Average Pooling followed by 1x1 Convolution
    branch4 = layers.AveragePooling2D(pool_size=(2, 2))(input_layer)
    branch4 = layers.Conv2D(32, (1, 1), activation='relu')(branch4)
    branch4 = layers.Dropout(0.5)(branch4)

    # Concatenate all branches
    concatenated = layers.concatenate([branch1, branch2, branch3, branch4])

    # Fully connected layers
    x = layers.Flatten()(concatenated)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(10, activation='softmax')(x)

    # Construct the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage
if __name__ == '__main__':
    model = dl_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))