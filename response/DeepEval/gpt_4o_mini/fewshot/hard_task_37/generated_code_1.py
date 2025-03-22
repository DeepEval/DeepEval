import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense, Concatenate

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Define the block
    def block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        return conv3

    # Create the two parallel branches
    block_output_1 = block(input_layer)
    block_output_2 = block(input_layer)

    # The parallel branch connecting through a convolutional layer
    parallel_branch = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine the outputs from all paths using addition
    combined_output = Add()([block_output_1, block_output_2, parallel_branch])

    # Concatenate outputs from the two blocks
    concatenated_output = Concatenate()([block_output_1, block_output_2])

    # Flatten the concatenated output and pass through a fully connected layer
    flatten_layer = Flatten()(concatenated_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model