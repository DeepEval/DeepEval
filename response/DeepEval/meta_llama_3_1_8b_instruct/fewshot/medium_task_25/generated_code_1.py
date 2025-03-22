import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, DepthwiseConv2D, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def branch1(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv

    def branch2(input_tensor):
        pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pool)
        return conv

    def branch3(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
        output_tensor = Concatenate()([conv2, conv3])
        return output_tensor

    def branch4(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv4 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
        output_tensor = Concatenate()([conv2, conv3, conv4])
        return output_tensor

    branch1_output = branch1(input_tensor=input_layer)
    branch2_output = branch2(input_tensor=input_layer)
    branch3_output = branch3(input_tensor=input_layer)
    branch4_output = branch4(input_tensor=input_layer)

    merged_output = Concatenate()([branch1_output, branch2_output, branch3_output, branch4_output])

    flatten = Flatten()(merged_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model