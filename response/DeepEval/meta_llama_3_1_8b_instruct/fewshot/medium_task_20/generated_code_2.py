import keras
from keras.layers import Input, Conv2D, Concatenate, MaxPooling2D, Flatten, Dense
from tensorflow.keras import layers

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def path_1(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        output_tensor = conv
        return output_tensor

    def path_2(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    def path_3(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
        output_tensor = Concatenate()([conv, conv2])
        return output_tensor

    def path_4(input_tensor):
        maxpool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(input_tensor)
        conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(maxpool)
        output_tensor = conv
        return output_tensor

    path1_output = path_1(input_layer)
    path2_output = path_2(input_layer)
    path3_output = path_3(input_layer)
    path4_output = path_4(input_layer)
    
    output_tensor = Concatenate()([path1_output, path2_output, path3_output, path4_output])
    flatten = Flatten()(output_tensor)
    dense = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model