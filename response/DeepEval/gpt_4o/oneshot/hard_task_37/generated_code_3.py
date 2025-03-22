import keras
from keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        # First path with three sequential convolutional layers
        conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1_1)
        conv1_3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1_2)

        # Second path with a single convolutional layer
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)

        # Combining the outputs of three paths using an addition operation
        added_output = Add()([conv1_1, conv1_2, conv1_3, conv2])

        return added_output

    # Two parallel branches using the same block
    branch1_output = block(input_layer)
    branch2_output = block(input_layer)

    # Concatenate the outputs from the two branches
    concatenated_output = Concatenate()([branch1_output, branch2_output])

    # Flatten and dense layers for final classification
    flatten_layer = Flatten()(concatenated_output)
    dense = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense)

    # Creating the Keras model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model