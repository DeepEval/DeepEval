import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # 1x1 initial convolutional layer
    conv1 = Conv2D(32, (1, 1), activation='relu')(input_layer)

    # First branch: local feature extraction
    branch1 = Conv2D(64, (3, 3), activation='relu')(conv1)
    branch1 = Conv2D(64, (3, 3), activation='relu')(branch1)

    # Second branch: downsampling with average pooling
    branch2 = MaxPooling2D((2, 2))(conv1)
    branch2 = Conv2D(64, (3, 3), activation='relu')(branch2)

    # Third branch: downsampling with average pooling
    branch3 = MaxPooling2D((2, 2))(conv1)
    branch3 = Conv2D(64, (3, 3), activation='relu')(branch3)

    # Upsampling with transposed convolutional layer
    branch3 = Conv2DTranspose(64, (3, 3), activation='relu')(branch3)

    # Concatenate outputs from all branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Refine output using 1x1 convolutional layer
    refined = Conv2D(64, (1, 1), activation='relu')(concatenated)

    # Flatten output
    flattened = Flatten()(refined)

    # Fully connected layers
    dense1 = Dense(128, activation='relu')(flattened)
    dense2 = Dense(64, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)

    # Create and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model