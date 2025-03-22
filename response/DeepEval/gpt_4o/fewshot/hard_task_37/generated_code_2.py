import keras
from keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        # Three sequential convolutional layers
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)

        # Main paths
        path1 = conv1
        path2 = conv2
        path3 = conv3

        # Parallel branch directly from the input
        parallel_branch = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)

        # Combine outputs using addition operation
        combined_output = Add()([path1, path2, path3, parallel_branch])
        return combined_output

    # Two parallel branches
    block1_output = block(input_layer)
    block2_output = block(input_layer)

    # Concatenate the outputs from both blocks
    concatenated = Concatenate()([block1_output, block2_output])

    # Flattening and fully connected layer
    flatten_layer = Flatten()(concatenated)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model