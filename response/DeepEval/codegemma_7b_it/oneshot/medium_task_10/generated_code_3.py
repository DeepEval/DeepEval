import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add, AveragePooling2D

def identity_block(input_tensor, filters):
    conv1 = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    conv2 = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    output_tensor = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
    return output_tensor

def residual_block(input_tensor, filters):
    conv1 = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    conv2 = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
    shortcut = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
    output_tensor = Add()([shortcut, conv3])
    return output_tensor

def global_branch(input_tensor):
    conv = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    avg_pool = AveragePooling2D(pool_size=(4, 4))(conv)
    return avg_pool

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    level1 = residual_block(input_tensor=conv, filters=16)

    level2 = residual_block(input_tensor=level1, filters=32)
    level2 = residual_block(input_tensor=level2, filters=32)

    level3 = global_branch(input_tensor=conv)

    concat = Concatenate()([level2, level3])
    flatten_layer = Flatten()(concat)
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model