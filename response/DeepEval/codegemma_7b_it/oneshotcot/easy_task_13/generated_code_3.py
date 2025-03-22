import keras
from keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    conv1_drop = Dropout(0.25)(conv1)

    conv2 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='valid', activation='relu')(conv1_drop)
    conv2_drop = Dropout(0.25)(conv2)

    conv3 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='valid', activation='relu')(conv2_drop)
    conv3_drop = Dropout(0.25)(conv3)

    conv4 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(conv3_drop)

    add_output = Add()([input_layer, conv4])

    flatten_layer = Flatten()(add_output)
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    dense1_drop = Dropout(0.5)(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense1_drop)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model