import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Dropout, Concatenate

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def branch_1(input_tensor):
        conv1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv1

    def branch_2(input_tensor):
        conv1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        return conv2

    def branch_3(input_tensor):
        conv1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        return conv3

    def branch_4(input_tensor):
        maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_tensor)
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(maxpool)
        return conv1

    branch1_output = branch_1(input_tensor=input_layer)
    branch2_output = branch_2(input_tensor=input_layer)
    branch3_output = branch_3(input_tensor=input_layer)
    branch4_output = branch_4(input_tensor=input_layer)

    concat_output = Concatenate()([branch1_output, branch2_output, branch3_output, branch4_output])
    flatten_layer = Flatten()(concat_output)
    drop_layer = Dropout(rate=0.2)(flatten_layer)
    dense1 = Dense(units=128, activation='relu')(drop_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    dense3 = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=dense3)

    return model