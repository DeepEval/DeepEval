import keras
from keras.layers import Input, Conv2D, BatchNormalization, Add, Flatten, Dense, Activation
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial Convolution
    initial_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    
    # Parallel Blocks
    def parallel_block(input_tensor):
        conv_layer = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_tensor)
        norm_layer = BatchNormalization()(conv_layer)
        activation_layer = Activation('relu')(norm_layer)
        return activation_layer

    block1_output = parallel_block(initial_conv)
    block2_output = parallel_block(initial_conv)
    block3_output = parallel_block(initial_conv)

    # Adding the outputs of the blocks with the initial convolution output
    added_output = Add()([initial_conv, block1_output, block2_output, block3_output])

    # Flattening and Fully Connected Layers
    flatten_layer = Flatten()(added_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model creation
    model = Model(inputs=input_layer, outputs=output_layer)

    return model