import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolution
    initial_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def block(input_tensor, filters):
        conv_layer = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        batch_norm = BatchNormalization()(conv_layer)
        activation = ReLU()(batch_norm)
        return activation

    # Three parallel blocks
    block1_output = block(input_tensor=initial_conv, filters=32)
    block2_output = block(input_tensor=initial_conv, filters=32)
    block3_output = block(input_tensor=initial_conv, filters=32)

    # Adding the outputs of the blocks with the initial convolution
    added_output = Add()([initial_conv, block1_output, block2_output, block3_output])

    # Flattening and fully connected layers
    flatten_layer = Flatten()(added_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Constructing the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model