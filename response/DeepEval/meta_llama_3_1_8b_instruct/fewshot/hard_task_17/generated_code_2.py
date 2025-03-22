import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Multiply, Reshape, Add, Dense, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        global_average_pooling = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=64, activation='relu')(global_average_pooling)
        dense2 = Dense(units=64, activation='relu')(dense1)
        reshape = Reshape(target_shape=(1, 64))(dense2)
        multiply = Multiply()([input_tensor, reshape])
        return multiply

    block1_output = block_1(input_layer)

    def block_2(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        max_pooling = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling)
        return conv2

    block2_output = block_2(input_tensor=block1_output)

    branch_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1_output)

    adding_layer = Add()([block2_output, branch_path])

    flatten_layer = Flatten()(adding_layer)
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model