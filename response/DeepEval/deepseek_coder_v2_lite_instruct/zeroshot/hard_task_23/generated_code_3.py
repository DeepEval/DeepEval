import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose, Flatten, Dense, concatenate

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # First branch: Local feature extraction
    x1 = Conv2D(32, (3, 3), activation='relu')(inputs)
    x1 = Conv2D(32, (3, 3), activation='relu')(x1)

    # Second branch: Downsampling and upsampling
    x2 = AveragePooling2D((2, 2))(inputs)
    x2 = Conv2D(64, (3, 3), activation='relu')(x2)
    x2 = Conv2DTranspose(64, (2, 2), strides=(2, 2))(x2)

    # Third branch: Downsampling and upsampling
    x3 = AveragePooling2D((2, 2))(inputs)
    x3 = Conv2D(64, (3, 3), activation='relu')(x3)
    x3 = Conv2DTranspose(64, (2, 2), strides=(2, 2))(x3)

    # Concatenate outputs from all branches
    combined = concatenate([x1, x2, x3])

    # Refine the consolidated output
    refined = Conv2D(128, (1, 1), activation='relu')(combined)

    # Flatten the output and feed into a fully connected layer
    flattened = Flatten()(refined)
    outputs = Dense(10, activation='softmax')(flattened)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
# model = dl_model()
# model.summary()