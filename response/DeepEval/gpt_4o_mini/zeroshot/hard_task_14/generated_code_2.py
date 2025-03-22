import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # Main path
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Fully connected layers in the main path
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)

    # Generate weights the size of the input channels
    weights = layers.Dense(3, activation='sigmoid')(x)

    # Reshape weights to match the input layer's shape
    reshaped_weights = layers.Reshape((1, 1, 3))(weights)

    # Element-wise multiplication with the original feature map
    weighted_input = layers.Multiply()([input_layer, reshaped_weights])

    # Branch path
    branch = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)

    # Combine both paths
    combined = layers.Add()([weighted_input, branch])

    # Fully connected layers after combining paths
    combined = layers.GlobalAveragePooling2D()(combined)
    combined = layers.Dense(128, activation='relu')(combined)
    combined = layers.Dense(64, activation='relu')(combined)
    output_layer = layers.Dense(10, activation='softmax')(combined)

    # Model construction
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of how to compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load and prepare CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Example of how to train the model
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))