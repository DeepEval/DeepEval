import keras
from keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1_1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(conv1_2)
    conv3 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(conv1_2)
    conv4 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(Add()([conv2, conv3]))

    dropout = Dropout(0.5)(conv4)
    flatten_layer = Flatten()(dropout)
    dense = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense)

    return model