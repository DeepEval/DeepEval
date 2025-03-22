from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the specialized block
    def block(filters, kernel_size, strides=1, padding='valid', kernel_initializer='he_normal', use_bias=False):
        return keras.Sequential([
            layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer, use_bias=use_bias),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(filters, 1, strides=1, padding='valid', kernel_initializer=kernel_initializer, use_bias=use_bias),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

    # Define the model
    inputs = keras.Input(shape=(28, 28, 1))
    x = block(32, (3, 3))
    x = layers.AveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    x = block(64, (3, 3))
    x = layers.AveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model