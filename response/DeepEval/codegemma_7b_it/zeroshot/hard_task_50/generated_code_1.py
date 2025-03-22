from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer
    inputs = keras.Input(shape=(32, 32, 3))

    # First block
    x = layers.MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = layers.MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(x)

    # Dropout to mitigate overfitting
    x = layers.Dropout(rate=0.4)(x)

    # Flatten and reshape for second block
    x = layers.Flatten()(x)
    x = layers.Reshape((4, -1))(x)

    # Second block
    x = layers.Lambda(lambda x: tf.split(x, 4, axis=-1))(x)

    # Separable convolutional layers
    x = [
        layers.SeparableConv2D(
            filters=128,
            kernel_size=(1, 1),
            padding='same'
        )(i) for i in x
    ]
    x = [
        layers.SeparableConv2D(
            filters=128,
            kernel_size=(3, 3),
            padding='same'
        )(i) for i in x
    ]
    x = [
        layers.SeparableConv2D(
            filters=128,
            kernel_size=(5, 5),
            padding='same'
        )(i) for i in x
    ]
    x = [
        layers.SeparableConv2D(
            filters=128,
            kernel_size=(7, 7),
            padding='same'
        )(i) for i in x
    ]

    # Concatenate outputs from second block
    x = layers.concatenate(x)

    # Classification layer
    outputs = layers.Dense(10, activation='softmax')(x)

    # Create model
    model = keras.Model(inputs, outputs)

    return model