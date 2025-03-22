import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Concatenate, AveragePooling2D, Lambda, Flatten, Dense

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Feature extraction path 1: 1x1 convolution
    fe1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs)
    fe1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(fe1)

    # Feature extraction path 2: sequence of convolutions
    fe2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(inputs)
    fe2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(fe2)
    fe2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(fe2)

    # Concatenate outputs from both paths
    merged = Concatenate()([fe1, fe2])

    # Additional 1x1 convolution to align dimensions with input image channel
    merged = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(merged)

    # Classification head
    outputs = Dense(units=10, activation='softmax')(merged)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Instantiate model
model = dl_model()

# Print model summary
model.summary()