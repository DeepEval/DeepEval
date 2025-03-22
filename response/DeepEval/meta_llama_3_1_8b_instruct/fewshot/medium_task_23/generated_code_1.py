import keras
from keras.layers import Input, Conv2D, Concatenate, AveragePooling2D, Dense, Flatten

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def path_1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv1

    def path_2(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
        return Concatenate()([conv1, conv2, conv3])

    def path_3(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv4 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(conv3)
        conv5 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
        return Concatenate()([conv1, conv4, conv5])

    def path_4(input_tensor):
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_tensor)
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(avg_pool)
        return conv1

    path1_output = path_1(input_tensor=input_layer)
    path2_output = path_2(input_tensor=input_layer)
    path3_output = path_3(input_tensor=input_layer)
    path4_output = path_4(input_tensor=input_layer)

    fused_output = Concatenate()([path1_output, path2_output, path3_output, path4_output])

    flatten = Flatten()(fused_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model