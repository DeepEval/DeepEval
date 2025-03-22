import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def conv_path1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv1

    def conv_path2(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        return conv2

    def conv_path3(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        return conv2

    def conv_path4(input_tensor):
        maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_tensor)
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(maxpool)
        return conv1

    path1_output = conv_path1(input_layer)
    path2_output = conv_path2(input_layer)
    path3_output = conv_path3(input_layer)
    path4_output = conv_path4(input_layer)

    concat_output = Concatenate()([path1_output, path2_output, path3_output, path4_output])
    flatten_output = Flatten()(concat_output)
    dense_output = Dense(units=128, activation='relu')(flatten_output)
    output_layer = Dense(units=10, activation='softmax')(dense_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model