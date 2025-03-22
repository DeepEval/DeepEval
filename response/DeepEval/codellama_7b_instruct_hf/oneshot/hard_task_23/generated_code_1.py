import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    conv1 = Conv2D(32, (1, 1), activation='relu')(input_layer)

    # First branch for local feature extraction
    branch1 = Conv2D(32, (3, 3), activation='relu')(conv1)
    branch1 = Conv2D(32, (3, 3), activation='relu')(branch1)

    # Second and third branches for downsampling and upsampling
    branch2 = MaxPooling2D((2, 2))(conv1)
    branch2 = Conv2D(32, (3, 3), activation='relu')(branch2)
    branch3 = MaxPooling2D((2, 2))(conv1)
    branch3 = Conv2D(32, (3, 3), activation='relu')(branch3)
    branch3 = Conv2D(32, (3, 3), activation='relu')(branch3)
    branch3 = Conv2D(32, (3, 3), activation='relu')(branch3)

    # Concatenate outputs of branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Refine output through 1x1 convolutional layer
    refined = Conv2D(32, (1, 1), activation='relu')(concatenated)

    # Batch normalization and flatten
    batch_norm = BatchNormalization()(refined)
    flatten = Flatten()(batch_norm)

    # Fully connected layers
    dense1 = Dense(128, activation='relu')(flatten)
    dense2 = Dense(64, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model