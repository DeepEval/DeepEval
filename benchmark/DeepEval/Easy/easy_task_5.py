from keras.models import Model
from keras.layers import Input, Conv2D, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28,28,1))

    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same',activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)

    flatten = Flatten()(conv3)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

