import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Initial convolution
    conv_initial = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    # Parallel blocks
    def block(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        batch_norm = BatchNormalization()(conv)
        relu = ReLU()(batch_norm)
        return relu

    block_1_output = block(input_tensor=conv_initial)
    block_2_output = block(input_tensor=conv_initial)
    block_3_output = block(input_tensor=conv_initial)

    # Add outputs of parallel blocks to initial convolution
    combined_output = Add()([conv_initial, block_1_output, block_2_output, block_3_output])

    # Flatten and fully connected layers
    flatten = Flatten()(combined_output)
    dense1 = Dense(units=64, activation='relu')(flatten)
    dense2 = Dense(units=32, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model