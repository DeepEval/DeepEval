import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Concatenate

def dl_model():
    # Define the input layer
    inputs = Input(shape=(32, 32, 3))

    # Branch 1: 1x1 convolution
    branch1 = Conv2D(32, (1, 1), activation='relu')(inputs)

    # Branch 2: 1x1 convolution followed by 3x3 convolution
    branch2 = Conv2D(32, (1, 1), activation='relu')(inputs)
    branch2 = Conv2D(32, (3, 3), activation='relu')(branch2)

    # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
    branch3 = Conv2D(32, (1, 1), activation='relu')(inputs)
    branch3 = Conv2D(32, (3, 3), activation='relu')(branch3)
    branch3 = Conv2D(32, (3, 3), activation='relu')(branch3)

    # Branch 4: Average pooling followed by 1x1 convolution
    branch4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    branch4 = Conv2D(32, (1, 1), activation='relu')(branch4)

    # Concatenate the outputs of all branches
    concatenated = Concatenate(axis=-1)([branch1, branch2, branch3, branch4])

    # Flatten the concatenated output
    flattened = Flatten()(concatenated)

    # Add dropout layers
    flattened = Dropout(0.5)(flattened)

    # Fully connected layers
    dense1 = Dense(128, activation='relu')(flattened)
    dense1 = Dropout(0.5)(dense1)
    dense2 = Dense(64, activation='relu')(dense1)
    dense2 = Dropout(0.5)(dense2)
    outputs = Dense(10, activation='softmax')(dense2)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model