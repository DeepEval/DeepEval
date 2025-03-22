import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose, Flatten, Dense, Concatenate

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # First branch: Local feature extraction
    x1 = Conv2D(32, (3, 3), activation='relu')(inputs)
    x1 = Conv2D(32, (3, 3), activation='relu')(x1)

    # Second branch: Downsampling followed by 3x3 convolution and upsampling
    x2 = AveragePooling2D((2, 2))(inputs)
    x2 = Conv2D(32, (3, 3), activation='relu')(x2)
    x2 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(x2)

    # Third branch: Similar to the second branch but without downsampling before upsampling
    x3 = AveragePooling2D((2, 2))(inputs)
    x3 = Conv2D(32, (3, 3), activation='relu')(x3)
    x3 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(x3)

    # Concatenate outputs of the three branches
    combined = Concatenate()([x1, x2, x3])

    # Refine the combined output with a 1x1 convolutional layer
    x4 = Conv2D(64, (1, 1), activation='relu')(combined)

    # Flatten and add a fully connected layer for classification
    x5 = Flatten()(x4)
    outputs = Dense(10, activation='softmax')(x5)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
# model = dl_model()
# model.summary()