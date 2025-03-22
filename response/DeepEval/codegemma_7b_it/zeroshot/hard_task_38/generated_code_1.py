from keras.models import Model
from keras.layers import (
    Input,
    BatchNormalization,
    Activation,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    concatenate,
)

def residual_block(x, filters):
    # Save the input for later use in the concatenation layer
    residual = x

    # First convolutional layer
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second convolutional layer
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    # Add the residual connection
    x = add([x, residual])
    x = Activation('relu')(x)

    return x

def dl_model():
    # Input layer
    inputs = Input(shape=(28, 28, 1))

    # Initial convolutional layer
    x = Conv2D(16, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # First pathway
    pathway1 = residual_block(x, 16)
    pathway1 = residual_block(pathway1, 16)
    pathway1 = residual_block(pathway1, 16)

    # Second pathway
    pathway2 = MaxPooling2D((2, 2))(x)
    pathway2 = residual_block(pathway2, 32)
    pathway2 = residual_block(pathway2, 32)
    pathway2 = residual_block(pathway2, 32)

    # Concatenate the outputs from both pathways
    merged = concatenate([pathway1, pathway2])

    # Fully connected layers
    x = Flatten()(merged)
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs, x)

    return model