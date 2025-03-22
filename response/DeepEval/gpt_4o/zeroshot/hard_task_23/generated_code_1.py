from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Conv2DTranspose, Concatenate, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial 1x1 convolution
    x = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)

    # Branch 1: Local feature extraction with two 3x3 convolutions
    branch1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    branch1 = Conv2D(64, (3, 3), activation='relu', padding='same')(branch1)

    # Branch 2: Average pooling, 3x3 convolution, and transposed convolution
    branch2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    branch2 = Conv2D(64, (3, 3), activation='relu', padding='same')(branch2)
    branch2 = Conv2DTranspose(64, (3, 3), activation='relu', strides=(2, 2), padding='same')(branch2)

    # Branch 3: Average pooling, 3x3 convolution, and transposed convolution
    branch3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    branch3 = Conv2D(64, (3, 3), activation='relu', padding='same')(branch3)
    branch3 = Conv2DTranspose(64, (3, 3), activation='relu', strides=(2, 2), padding='same')(branch3)

    # Concatenate branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Refinement with a 1x1 convolution
    refined = Conv2D(64, (1, 1), activation='relu', padding='same')(concatenated)

    # Flatten and fully connected layer for classification
    flat = Flatten()(refined)
    output = Dense(10, activation='softmax')(flat)

    # Create the model
    model = Model(inputs=input_layer, outputs=output)

    return model

# Example usage
model = dl_model()
model.summary()