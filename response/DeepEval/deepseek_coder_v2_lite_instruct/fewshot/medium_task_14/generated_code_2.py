import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    def block(input_tensor, filters):
        x = Conv2D(filters, kernel_size=(3, 3), padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    # First block
    block1_output = block(input_tensor=input_layer, filters=32)
    block1_output = block(input_tensor=block1_output, filters=32)

    # Second block
    block2_output = block(input_tensor=block1_output, filters=64)
    block2_output = block(input_tensor=block2_output, filters=64)

    # Third block
    block3_output = block(input_tensor=block2_output, filters=128)
    block3_output = block(input_tensor=block3_output, filters=128)

    # Parallel branch
    parallel_branch = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(input_layer)
    parallel_branch = BatchNormalization()(parallel_branch)
    parallel_branch = ReLU()(parallel_branch)

    # Sum the outputs from all paths
    added_output = Add()([block1_output, block2_output, block3_output, parallel_branch])

    # Flatten and pass through fully connected layers
    flattened_output = Flatten()(added_output)
    dense1 = Dense(units=256, activation='relu')(flattened_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model