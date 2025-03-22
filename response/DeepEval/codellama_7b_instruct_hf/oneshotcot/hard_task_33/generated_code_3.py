import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Define the three branches
    branch1 = input_layer
    branch2 = input_layer
    branch3 = input_layer

    # Define the same block for each branch
    for i in range(3):
        # Elevate the dimension through a 1x1 convolutional layer
        branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch1)
        # Extract features through a 3x3 depthwise separable convolutional layer
        branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
        # Reduce the dimension through a 1x1 convolutional layer
        branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch1)

        # Elevate the dimension through a 1x1 convolutional layer
        branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch2)
        # Extract features through a 3x3 depthwise separable convolutional layer
        branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
        # Reduce the dimension through a 1x1 convolutional layer
        branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch2)

        # Elevate the dimension through a 1x1 convolutional layer
        branch3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch3)
        # Extract features through a 3x3 depthwise separable convolutional layer
        branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
        # Reduce the dimension through a 1x1 convolutional layer
        branch3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch3)

    # Concatenate the outputs from the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Flatten the concatenated output
    flattened = Flatten()(concatenated)

    # Pass the flattened output through a fully connected layer to generate classification probabilities
    output_layer = Dense(units=10, activation='softmax')(flattened)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model