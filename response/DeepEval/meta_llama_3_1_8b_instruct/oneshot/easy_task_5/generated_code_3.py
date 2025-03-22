import keras
from keras.layers import Input, Conv2D, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1x1)
    conv1x1_restore = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv3x3)
    flatten_layer = Flatten()(conv1x1_restore)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model