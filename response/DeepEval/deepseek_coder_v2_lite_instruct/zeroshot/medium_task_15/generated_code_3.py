import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate, AveragePooling2D, Flatten

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Convolutional layer
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Feature maps compression and reshaping
    y = Conv2D(64, (1, 1), padding='same')(x)
    y = BatchNormalization()(y)
    y = ReLU()(y)
    y = GlobalAveragePooling2D()(y)
    y = Dense(128, activation='relu')(y)  # Fully connected layer

    # Reshape and multiply with initial features
    y = Reshape((1, 1, 128))(y)
    weighted_features = Multiply()([x, y])

    # Concatenate with input layer
    z = Concatenate()([inputs, weighted_features])

    # Dimensionality reduction and downsampling
    z = Conv2D(64, (1, 1), padding='same')(z)
    z = BatchNormalization()(z)
    z = ReLU()(z)
    z = AveragePooling2D((2, 2))(z)

    # Flatten and fully connected layer
    z = Flatten()(z)
    outputs = Dense(10, activation='softmax')(z)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
# model = dl_model()
# model.summary()