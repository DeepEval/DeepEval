import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Reshape, Multiply, Add

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        gap = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=input_tensor.shape[-1], activation='relu')(gap)
        dense2 = Dense(units=input_tensor.shape[-1], activation='relu')(dense1)
        dense3 = Dense(units=input_tensor.shape[-1], activation='relu')(dense2)
        reshape = Reshape((1, 1, input_tensor.shape[-1]))(dense3)
        weighted_output = Multiply()([reshape, input_tensor])
        return weighted_output

    def block_2(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
        return max_pooling

    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=input_layer)

    branch_output = block_2(input_tensor=block1_output)
    main_output = block2_output
    fused_output = Add()([main_output, branch_output])

    flatten = Flatten()(fused_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model