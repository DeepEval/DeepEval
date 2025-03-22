import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    input_shape = (32, 32, 3)
    input_layer = Input(shape=input_shape)
    split_layer = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    conv1_layer = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(split_layer[0])
    conv2_layer = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(split_layer[1])
    conv3_layer = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(split_layer[2])
    pool1_layer = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1_layer)
    pool2_layer = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2_layer)
    pool3_layer = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3_layer)
    concat_layer = Concatenate(axis=3)([pool1_layer, pool2_layer, pool3_layer])
    flatten_layer = Flatten()(concat_layer)
    dense1_layer = Dense(units=128, activation='relu')(flatten_layer)
    dense2_layer = Dense(units=10, activation='softmax')(dense1_layer)
    model = Model(inputs=input_layer, outputs=dense2_layer)
    return model