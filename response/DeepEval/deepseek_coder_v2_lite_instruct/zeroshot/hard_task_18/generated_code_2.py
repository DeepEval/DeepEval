import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten, Add, Multiply

def dl_model():
    # Define the input shape
    inputs = Input(shape=(32, 32, 3))

    # First block
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = AveragePooling2D((2, 2))(x)
    shortcut = x

    # Main path
    y = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    y = Conv2D(64, (3, 3), activation='relu', padding='same')(y)
    y = AveragePooling2D((2, 2))(y)

    # Addition of shortcut and main path
    x = Add()([shortcut, y])

    # Second block
    # Global Average Pooling
    z = tf.reduce_mean(x, axis=(1, 2))
    z = Dense(64, activation='relu')(z)
    z = Dense(64, activation='relu')(z)

    # Generate channel weights
    weights = Dense(64, activation='softmax')(z)

    # Reshape weights to match input shape
    weights = tf.reshape(weights, (-1, 64, 1, 1))

    # Multiply weights with input
    x = Multiply()([x, weights])

    # Flatten and pass through final fully connected layer
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model