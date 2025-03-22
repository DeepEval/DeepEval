import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Flatten, Dense, Add

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    initial_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def block(input_tensor, filters, kernel_size):
        conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding='same')(input_tensor)
        batch_norm = BatchNormalization()(conv)
        activation = ReLU()(batch_norm)
        return activation

    block1 = block(initial_conv, filters=32, kernel_size=(3, 3))
    block2 = block(initial_conv, filters=32, kernel_size=(3, 3))
    block3 = block(initial_conv, filters=32, kernel_size=(3, 3))

    added_output = Add()([initial_conv, block1, block2, block3])
    flatten_layer = Flatten()(added_output)
    
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model