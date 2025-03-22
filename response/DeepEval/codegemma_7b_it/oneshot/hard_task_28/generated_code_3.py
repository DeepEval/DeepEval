import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Conv2DTranspose, LayerNormalization, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def residual_block(x, filters, kernel_size, strides):
    conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', activation='relu')(x)
    bn = BatchNormalization()(conv)
    return bn

def main_path(input_tensor):
    conv = DepthwiseConv2D(kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu')(input_tensor)
    bn = BatchNormalization()(conv)
    return bn

def branch_path(input_tensor):
    conv = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    bn = BatchNormalization()(conv)
    return bn

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    main_path_output = main_path(input_tensor)
    main_path_output = residual_block(main_path_output, filters=64, kernel_size=(1, 1), strides=(2, 2))
    main_path_output = residual_block(main_path_output, filters=64, kernel_size=(1, 1), strides=(1, 1))

    branch_path_output = branch_path(input_tensor)

    combined_output = Concatenate()([main_path_output, branch_path_output])
    bn = BatchNormalization()(combined_output)

    flatten_layer = Flatten()(bn)
    dense1 = Dense(units=512, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model