import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Flatten, Dense

def dl_model():
    # Input layer
    input_img = Input(shape=(32, 32, 3)) 

    # Initial 1x1 convolution
    x = Conv2D(32, (1, 1), activation='relu')(input_img)

    # Branch 1
    branch1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    # Branch 2
    branch2 = MaxPooling2D((2, 2))(x)
    branch2 = Conv2D(128, (3, 3), activation='relu', padding='same')(branch2)
    branch2 = UpSampling2D((2, 2))(branch2)

    # Branch 3
    branch3 = MaxPooling2D((2, 2))(x)
    branch3 = Conv2D(128, (3, 3), activation='relu', padding='same')(branch3)
    branch3 = UpSampling2D((2, 2))(branch3)

    # Concatenate branches
    combined = concatenate([branch1, branch2, branch3], axis=3)

    # Final 1x1 convolution
    x = Conv2D(128, (1, 1), activation='relu')(combined)

    # Flatten and fully connected layers
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_img, outputs=output)

    return model

# Example usage:
model = dl_model()
model.summary()