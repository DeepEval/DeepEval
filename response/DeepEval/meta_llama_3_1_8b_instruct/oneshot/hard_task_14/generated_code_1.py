import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Add
from keras.regularizers import l2

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    global_avg_pool = GlobalAveragePooling2D()(input_layer)

    weights = Dense(units=3, activation='relu')(global_avg_pool)
    weights = Reshape((3, 1, 1))(weights)
    element_wise_mul = Multiply()([input_layer, weights])

    branch_conv = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    element_wise_add = Add()([element_wise_mul, branch_conv])

    def block(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3)
        output_tensor = Concatenate()([conv1, conv2, conv3, maxpool])

        return output_tensor

    block_output = block(element_wise_add)
    bath_norm = BatchNormalization()(block_output)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model