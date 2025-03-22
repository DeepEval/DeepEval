import keras
from keras.layers import Input, Lambda, Conv2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    # Branch path
    input_branch = Input(shape=(32, 32, 3))
    conv_branch = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_branch)
    conv_branch = BatchNormalization()(conv_branch)
    conv_branch = Activation('relu')(conv_branch)

    # Main path
    input_main = Input(shape=(32, 32, 3))
    split = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(input_main)
    feature_extract_group = split[0]
    input_group_2 = split[1]
    input_group_3 = split[2]

    conv_main_1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(feature_extract_group)
    conv_main_2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_group_2)
    conv_main_3 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_group_3)

    concat = Concatenate(axis=-1)([conv_main_1, conv_main_2, conv_main_3])
    concat = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(concat)
    concat = BatchNormalization()(concat)
    concat = Activation('relu')(concat)

    # Output layer
    flatten = Flatten()(concat)
    output = Dense(units=10, activation='softmax')(flatten)

    model = Model(inputs=[input_branch, input_main], outputs=output)

    return model

model = dl_model()