import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # Main path
    main_path = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    main_path = layers.MaxPooling2D((2, 2))(main_path)
    main_path = layers.Conv2D(64, (3, 3), activation='relu')(main_path)
    main_path = layers.MaxPooling2D((2, 2))(main_path)
    main_path = layers.Conv2D(128, (3, 3), activation='relu')(main_path)
    main_path = layers.GlobalAveragePooling2D()(main_path)

    # Fully connected layers in the main path
    main_path = layers.Dense(256, activation='relu')(main_path)
    main_path = layers.Dense(32 * 32 * 3, activation='sigmoid')(main_path)  # output shape matches input

    # Reshape to match the input layer's shape
    main_path = layers.Reshape((32, 32, 3))(main_path)

    # Element-wise multiplication with the input feature map
    main_path = layers.multiply([main_path, input_layer])

    # Branch path (direct connection from input)
    branch_path = input_layer

    # Combine both paths
    combined = layers.add([main_path, branch_path])

    # Fully connected layers after combining
    combined = layers.GlobalAveragePooling2D()(combined)
    combined = layers.Dense(256, activation='relu')(combined)
    output_layer = layers.Dense(10, activation='softmax')(combined)  # 10 classes for CIFAR-10

    # Create model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example to compile the model
if __name__ == "__main__":
    model = dl_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Fit the model
    model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))