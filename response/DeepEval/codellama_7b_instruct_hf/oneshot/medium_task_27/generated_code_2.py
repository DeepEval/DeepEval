import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model
from keras.applications.vgg16 import VGG16


def dl_model():
    input_layer = Input(shape=(32, 32, 3))


    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    branch1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch1)


    branch2 = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch2)


    concat_layer = Concatenate()([branch1, branch2])


    batch_norm_layer = BatchNormalization()(concat_layer)


    flatten_layer = Flatten()(batch_norm_layer)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)


    model = Model(inputs=input_layer, outputs=dense2)
    return model