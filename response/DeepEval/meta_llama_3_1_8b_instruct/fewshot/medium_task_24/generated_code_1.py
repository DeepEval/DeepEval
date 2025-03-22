import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Dropout, Dense, Flatten

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def branch_1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        output_tensor = Concatenate()([conv1, conv2])
        output_tensor = Dropout(0.2)(output_tensor)
        return output_tensor

    def branch_2(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 7), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(7, 1), strides=(7, 1), padding='same', activation='relu')(conv1)
        conv4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        output_tensor = Concatenate()([conv1, conv2, conv3, conv4])
        output_tensor = Dropout(0.2)(output_tensor)
        return output_tensor

    def branch_3(input_tensor):
        output_tensor = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(input_tensor)
        output_tensor = Dropout(0.2)(output_tensor)
        return output_tensor

    branch1_output = branch_1(input_tensor=input_layer)
    branch2_output = branch_2(input_tensor=input_layer)
    branch3_output = branch_3(input_tensor=input_layer)

    concatenated = Concatenate()([branch1_output, branch2_output, branch3_output])

    flatten_layer = Flatten()(concatenated)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model