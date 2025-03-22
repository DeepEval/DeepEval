import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, BatchNormalization, Flatten, Dense, Multiply, GlobalAveragePooling2D
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First block
    def first_block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
        add1 = Add()([input_tensor, conv2])
        avg_pool = AveragePooling2D(pool_size=(2, 2))(add1)
        return avg_pool

    first_block_output = first_block(input_layer)

    # Second block
    def second_block(input_tensor):
        global_avg_pool = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=128, activation='relu')(global_avg_pool)
        dense2 = Dense(units=64, activation='relu')(dense1)
        reshape_weights = Dense(units=first_block_output.shape[1] * first_block_output.shape[2] * first_block_output.shape[3], activation='relu')(dense2)
        reshape_weights = keras.backend.reshape(reshape_weights, (1, first_block_output.shape[1], first_block_output.shape[2], first_block_output.shape[3]))
        multiply_weights = Multiply()([first_block_output, reshape_weights])
        return multiply_weights

    second_block_output = second_block(first_block_output)

    # Flatten and final classification layer
    flatten_layer = Flatten()(second_block_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct and return the model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model