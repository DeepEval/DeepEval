from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras import backend as K

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize pixel values
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Define the model
    inputs = keras.Input(shape=(32, 32, 3))

    # Conv2D block
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation='relu')(x)

    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Two fully connected layers
    weights = layers.Dense(128, use_bias=False)(x)
    weights = layers.Reshape((128, 1))(weights)

    # Element-wise multiplication
    x = layers.Multiply()([inputs, weights])

    # Flatten and output layer
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Usage
model = dl_model()
model.summary()