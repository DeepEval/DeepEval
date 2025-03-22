import keras
from keras.layers import Input, DepthwiseSeparableConv2D, BatchNormalization, Flatten, Dense, Dropout

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Define the first branch
    branch1 = DepthwiseSeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid')(input_layer)
    branch1 = BatchNormalization()(branch1)
    branch1 = Dropout(rate=0.2)(branch1)
    branch1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same')(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = Dropout(rate=0.2)(branch1)

    # Define the second branch
    branch2 = DepthwiseSeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid')(input_layer)
    branch2 = BatchNormalization()(branch2)
    branch2 = Dropout(rate=0.2)(branch2)
    branch2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same')(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = Dropout(rate=0.2)(branch2)

    # Define the third branch
    branch3 = DepthwiseSeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid')(input_layer)
    branch3 = BatchNormalization()(branch3)
    branch3 = Dropout(rate=0.2)(branch3)
    branch3 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same')(branch3)
    branch3 = BatchNormalization()(branch3)
    branch3 = Dropout(rate=0.2)(branch3)

    # Concatenate the outputs of the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Flatten the concatenated output
    flattened = Flatten()(concatenated)

    # Add a fully connected layer
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Add a final fully connected layer for the output
    output = Dense(units=10, activation='softmax')(dense2)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output)

    return model