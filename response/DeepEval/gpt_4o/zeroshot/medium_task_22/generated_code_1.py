import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: 3x3 convolutions
    branch1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    
    # Branch 2: 1x1 convolution followed by two 3x3 convolutions
    branch2 = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)
    branch2 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch2)
    branch2 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch2)

    # Branch 3: Max pooling
    branch3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    # Concatenate the branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Flatten and add fully connected layers
    flattened = Flatten()(concatenated)
    fc1 = Dense(128, activation='relu')(flattened)
    fc2 = Dense(64, activation='relu')(fc1)

    # Output layer with 10 classes (for CIFAR-10)
    output_layer = Dense(10, activation='softmax')(fc2)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of how to create the model
model = dl_model()
model.summary()