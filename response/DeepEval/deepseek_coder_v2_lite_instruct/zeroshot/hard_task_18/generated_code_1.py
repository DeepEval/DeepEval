import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten, Add, Multiply, GlobalAveragePooling2D

def dl_model():
    # Define the input shape
    inputs = Input(shape=(32, 32, 3))

    # First block
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = AveragePooling2D((2, 2))(x)
    x = Add()([inputs, x])

    # Second block
    y = GlobalAveragePooling2D()(x)
    y = Dense(128, activation='relu')(y)
    y = Dense(128, activation='relu')(y)
    y = Dense(32, activation='relu')(y)  # Assuming the number of channels in the output of the first block is 32
    y = tf.expand_dims(y, axis=1)
    y = tf.expand_dims(y, axis=1)
    y = Multiply()([x, y])

    # Final classification layer
    z = Flatten()(y)
    outputs = Dense(10, activation='softmax')(z)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model