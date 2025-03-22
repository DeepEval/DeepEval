import keras
from keras.layers import Input, Conv2D, Add, Dropout, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_1_dropout = Dropout(0.2)(conv1_1)
    conv1_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1_1_dropout)
    conv1_2_dropout = Dropout(0.2)(conv1_2)

    conv2 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(conv1_2_dropout)
    conv2_dropout = Dropout(0.2)(conv2)
    conv3 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(conv1_2_dropout)
    conv3_dropout = Dropout(0.2)(conv3)

    conv4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2_dropout + conv3_dropout)
    conv4_dropout = Dropout(0.2)(conv4)

    add_layer = Add()([input_layer, conv4_dropout])
    flatten_layer = Flatten()(add_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model