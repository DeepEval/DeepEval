import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, GlobalAveragePooling2D, Dense, Multiply, Flatten
from keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # First block
    def first_block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
        add_path = Add()([input_tensor, conv2])
        avg_pool = AveragePooling2D(pool_size=(2, 2))(add_path)
        return avg_pool

    # Apply the first block to the input layer
    first_block_output = first_block(input_layer)

    # Second block
    def second_block(input_tensor):
        global_avg_pool = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=128, activation='relu')(global_avg_pool)
        dense2 = Dense(units=64, activation='relu')(dense1)
        reshape_weights = Dense(units=32, activation='relu')(dense2)
        reshape_weights = reshape_weights.reshape((1, 32))
        reshape_weights = reshape_weights.reshape((32, 1, 1, 32))
        reshape_weights = keras.backend.repeat_elements(reshape_weights, 32, axis=0)
        reshape_weights = keras.backend.repeat_elements(reshape_weights, 32, axis=1)
        reshape_weights = keras.backend.repeat_elements(reshape_weights, 3, axis=2)
        reshape_weights = keras.backend.repeat_elements(reshape_weights, 3, axis=3)
        multiply_weights = Multiply()([input_tensor, reshape_weights])
        return multiply_weights

    # Apply the second block to the first block's output
    second_block_output = second_block(first_block_output)

    # Flatten the output and add the final dense layer for classification
    flatten_layer = Flatten()(second_block_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model