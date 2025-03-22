import keras
from keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        return conv3

    # First branch
    branch1_output = block(input_layer)
    
    # Second branch
    branch2_output = block(input_layer)

    # Parallel branch with a convolutional layer
    parallel_branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine the outputs of the branches using an addition operation
    combined_output = Add()([branch1_output, branch2_output, parallel_branch])

    # Concatenate the outputs from both blocks
    concatenated_output = Concatenate()([branch1_output, branch2_output])

    # Flatten the concatenated output
    flatten_layer = Flatten()(concatenated_output)

    # Fully connected layer
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)

    # Output layer for classification
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model