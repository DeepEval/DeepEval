import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape

def block_1(input_tensor):
    avg_pool_1x1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
    avg_pool_2x2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
    avg_pool_4x4 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
    avg_outputs = Concatenate()([avg_pool_1x1, avg_pool_2x2, avg_pool_4x4])
    return avg_outputs

def block_2(input_tensor):
    conv_branch_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    conv_branch_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)

    conv_branch_1x7_7x1 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    conv_branch_7x1_1x7 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv_branch_1x7_7x1)

    conv_branch_3x3_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
    conv_branch_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_branch_3x3_pool)

    concat_outputs = Concatenate()([conv_branch_1x1, conv_branch_3x3, conv_branch_7x1_1x7])
    return concat_outputs

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    block_1_output = block_1(input_tensor=input_layer)
    block_1_flatten = Flatten()(block_1_output)

    block_2_input = Reshape((1, 1, -1))(block_1_flatten)
    block_2_output = block_2(input_tensor=block_2_input)

    dense1 = Dense(units=128, activation='relu')(block_2_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model