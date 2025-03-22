import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, BatchNormalization, ReLU

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        batch_norm = BatchNormalization()(input_tensor)
        relu = ReLU()(batch_norm)
        conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu)
        return conv

    block1_output = block(input_tensor=input_layer)
    block2_output = block(input_tensor=block1_output)
    block3_output = block(input_tensor=block2_output)

    block4_output = block(input_tensor=input_layer)
    block5_output = block(input_tensor=block4_output)
    block6_output = block(input_tensor=block5_output)

    merge_layer = Add()([block1_output, block2_output, block3_output, block4_output, block5_output, block6_output])
    flatten_layer = Flatten()(merge_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model