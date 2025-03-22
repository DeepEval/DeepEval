import keras
from keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        # Three sequential convolutional layers
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv2)

        # Additional parallel path directly from the input
        parallel_conv = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)

        # Combine the outputs using addition
        added_output = Add()([conv1, conv2, conv3, parallel_conv])

        return added_output

    # Two parallel branches with the same block
    block1_output = block(input_layer)
    block2_output = block(input_layer)

    # Concatenate the outputs of the two blocks
    concatenated_output = Concatenate()([block1_output, block2_output])

    # Flatten and add a fully connected layer
    flattened_output = Flatten()(concatenated_output)
    dense_layer = Dense(units=128, activation='relu')(flattened_output)

    # Final classification layer
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model