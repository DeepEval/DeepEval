import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Concatenate, Dense

def residual_block(inputs):
    x = Conv2D(64, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Add()([x, inputs])
    return x

def dl_model():
    # Input layer
    inputs = Input(shape=(28, 28, 1))

    # Pathway 1
    x1 = Conv2D(64, (3, 3), padding='same')(inputs)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    for _ in range(3):
        x1 = residual_block(x1)

    # Pathway 2
    x2 = Conv2D(64, (3, 3), padding='same')(inputs)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)
    for _ in range(3):
        x2 = residual_block(x2)

    # Concatenate outputs from both pathways
    combined = Concatenate(axis=-1)([x1, x2])

    # Flatten and add dense layers
    x = tf.keras.layers.Flatten()(combined)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
model = dl_model()
model.summary()