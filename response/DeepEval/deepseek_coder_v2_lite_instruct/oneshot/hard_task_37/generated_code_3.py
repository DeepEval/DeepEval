import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        # First convolutional layer
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Second convolutional layer
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        # Third convolutional layer
        conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)

        # Path 1: Direct convolutional layer
        path1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)

        # Combine outputs from all paths
        combined = Add()([conv3, path1])

        return combined

    # Apply the block to the input
    block1_output = block(input_tensor=input_layer)
    block2_output = block(input_tensor=block1_output)

    # Concatenate outputs from the two blocks
    concatenated = Concatenate()([block1_output, block2_output])

    # Flatten the result
    flatten_layer = Flatten()(concatenated)

    # Fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model