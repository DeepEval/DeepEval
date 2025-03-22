import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dense, Flatten

def dl_model():
    # Define the input layer
    inputs = Input(shape=(32, 32, 3))

    # Initial 1x1 convolutional layer
    x = Conv2D(32, (1, 1), activation='relu')(inputs)

    # Branch 1: Local feature extraction
    branch1 = Conv2D(32, (3, 3), activation='relu')(x)

    # Branch 2: Sequential layers including max pooling, 3x3 convolution, and upsampling
    branch2 = MaxPooling2D((2, 2))(x)
    branch2 = Conv2D(64, (3, 3), activation='relu')(branch2)
    branch2 = UpSampling2D((2, 2))(branch2)

    # Branch 3: Sequential layers including max pooling, 3x3 convolution, and upsampling
    branch3 = MaxPooling2D((2, 2))(x)
    branch3 = Conv2D(64, (3, 3), activation='relu')(branch3)
    branch3 = UpSampling2D((2, 2))(branch3)

    # Concatenate outputs of all branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Additional 1x1 convolutional layer
    x = Conv2D(64, (1, 1), activation='relu')(concatenated)

    # Flatten the output and pass through fully connected layers
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
# model = dl_model()
# model.summary()