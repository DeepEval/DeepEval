import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Define the first 1x1 convolutional layer
    conv1 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    conv1 = Dropout(0.2)(conv1)

    # Define the second 1x1 convolutional layer
    conv2 = Conv2D(32, (1, 1), activation='relu')(conv1)
    conv2 = Dropout(0.2)(conv2)

    # Define the third 3x1 convolutional layer
    conv3 = Conv2D(32, (3, 1), activation='relu')(conv2)
    conv3 = Dropout(0.2)(conv3)

    # Define the fourth 1x3 convolutional layer
    conv4 = Conv2D(32, (1, 3), activation='relu')(conv3)
    conv4 = Dropout(0.2)(conv4)

    # Define the channel-wise add operation
    add = Concatenate(axis=1)([conv1, conv2, conv3, conv4])

    # Define the batch normalization layer
    batch_norm = BatchNormalization()(add)

    # Define the flattening layer
    flatten = Flatten()(batch_norm)

    # Define the fully connected layers
    dense1 = Dense(64, activation='relu')(flatten)
    dense1 = Dropout(0.2)(dense1)
    dense2 = Dense(32, activation='relu')(dense1)
    dense2 = Dropout(0.2)(dense2)
    output_layer = Dense(10, activation='softmax')(dense2)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model