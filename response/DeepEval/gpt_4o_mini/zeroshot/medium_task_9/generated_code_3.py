import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def basic_block(x, filters):
    # Main path
    conv = layers.Conv2D(filters, (3, 3), padding='same')(x)
    bn = layers.BatchNormalization()(conv)
    relu = layers.Activation('relu')(bn)

    # Branch path
    branch = layers.Conv2D(filters, (1, 1), padding='same')(x)

    # Feature fusion through addition
    output = layers.add([relu, branch])
    return output

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 image dimensions
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolutional layer to reduce dimensionality to 16
    x = layers.Conv2D(16, (3, 3), padding='same')(inputs)

    # Two consecutive basic blocks
    x = basic_block(x, 16)
    x = basic_block(x, 16)

    # Additional branch layer
    branch = layers.Conv2D(16, (1, 1), padding='same')(inputs)

    # Combine outputs
    x = layers.add([x, branch])

    # Average pooling to downsample
    x = layers.AveragePooling2D(pool_size=(8, 8))(x)

    # Flatten the feature map
    x = layers.Flatten()(x)

    # Fully connected layer for classification
    outputs = layers.Dense(10, activation='softmax')(x)

    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example of how to use the model
if __name__ == "__main__":
    model = dl_model()
    model.summary()  # Print the model summary

    # Load and preprocess the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model (example: train for 10 epochs)
    model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))