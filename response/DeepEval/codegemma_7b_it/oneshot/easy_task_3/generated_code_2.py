import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)
        return maxpool

    block1_output = block(input_tensor)
    block2_output = block(block1_output)

    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2_output)
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
    conv5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv4)
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv5)

    flatten_layer = Flatten()(maxpool2)

    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model