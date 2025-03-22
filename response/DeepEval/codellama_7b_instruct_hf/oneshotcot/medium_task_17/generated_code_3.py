import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    input_tensor = input_layer.permute((0, 2, 3, 1))
    input_tensor = input_tensor.reshape((-1, 3, 32, 32))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_tensor)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(max_pooling1)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(max_pooling2)
    max_pooling3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3)
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(max_pooling3)
    max_pooling4 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv4)
    conv5 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(max_pooling4)
    max_pooling5 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv5)
    flatten = Flatten()(max_pooling5)
    dense1 = Dense(units=1024, activation='relu')(flatten)
    dense2 = Dense(units=512, activation='relu')(dense1)
    dense3 = Dense(units=10, activation='softmax')(dense2)
    model = Model(inputs=input_layer, outputs=dense3)
    return model