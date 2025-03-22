import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 RGB
    inputs = layers.Input(shape=input_shape)

    # Main pathway
    # Path 1: 1x1 Convolution
    path1 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(inputs)

    # Path 2: Parallel convolutions
    path2_1 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(inputs)
    path2_2 = layers.Conv2D(32, (1, 3), padding='same', activation='relu')(inputs)
    path2_3 = layers.Conv2D(32, (3, 1), padding='same', activation='relu')(inputs)

    # Concatenate outputs from path 2
    path2 = layers.concatenate([path2_1, path2_2, path2_3])

    # Concatenate path1 and path2
    merged = layers.concatenate([path1, path2])

    # Final 1x1 Convolution to reduce channels
    output = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(merged)

    # Add a direct connection from inputs
    added = layers.add([output, inputs])

    # Flattening before fully connected layers
    flatten = layers.Flatten()(added)

    # Fully connected layers
    dense1 = layers.Dense(128, activation='relu')(flatten)
    dense2 = layers.Dense(10, activation='softmax')(dense1)  # CIFAR-10 has 10 classes

    # Create the model
    model = models.Model(inputs=inputs, outputs=dense2)

    return model

# Example of using the model
if __name__ == "__main__":
    model = dl_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Train the model
    model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))