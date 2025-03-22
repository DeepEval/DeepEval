import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Concatenate

def dl_model():
    input_layer = Input(shape=(32,32,3))

    def branch_1(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        drop1 = Dropout(rate=0.2)(conv2)
        return drop1

    def branch_2(input_tensor):
        conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv4 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(conv3)
        conv5 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv4)
        conv6 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv5)
        drop2 = Dropout(rate=0.2)(conv6)
        return drop2

    def branch_3(input_tensor):
        maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_tensor)
        drop3 = Dropout(rate=0.2)(maxpool)
        return drop3

    branch1_output = branch_1(input_tensor=input_layer)
    branch2_output = branch_2(input_tensor=input_layer)
    branch3_output = branch_3(input_tensor=input_layer)

    concat_layer = Concatenate()([branch1_output, branch2_output, branch3_output])

    flatten_layer = Flatten()(concat_layer)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model