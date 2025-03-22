import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense, Concatenate

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        return conv3

    # Create the first branch
    block_output_1 = block(input_layer)

    # Create the second branch with the same block
    block_output_2 = block(input_layer)

    # Add the outputs from the two blocks
    added_output = Add()([block_output_1, block_output_2])

    # Create a parallel branch with a convolutional layer
    parallel_branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine the outputs from the added and parallel branches
    combined_output = Add()([added_output, parallel_branch])

    # Flatten and fully connected layer for classification
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model