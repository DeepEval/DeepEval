import keras
from keras.layers import Input, Conv2D, Concatenate, BatchNormalization, Dropout, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def branch_1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        dropout1 = Dropout(0.2)(conv1)
        return dropout1

    def branch_2(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        dropout2 = Dropout(0.2)(conv2)
        return dropout2

    def branch_3(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        dropout3 = Dropout(0.2)(conv3)
        return dropout3

    def branch_4(input_tensor):
        max_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(max_pool)
        dropout4 = Dropout(0.2)(conv1)
        return dropout4

    branch1_output = branch_1(input_tensor=input_layer)
    branch2_output = branch_2(input_tensor=input_layer)
    branch3_output = branch_3(input_tensor=input_layer)
    branch4_output = branch_4(input_tensor=input_layer)

    concatenated_output = Concatenate()([branch1_output, branch2_output, branch3_output, branch4_output])

    batch_norm = BatchNormalization()(concatenated_output)

    flatten_layer = Flatten()(batch_norm)

    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dropout5 = Dropout(0.2)(dense1)

    dense2 = Dense(units=64, activation='relu')(dropout5)
    dropout6 = Dropout(0.2)(dense2)

    output_layer = Dense(units=10, activation='softmax')(dropout6)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model