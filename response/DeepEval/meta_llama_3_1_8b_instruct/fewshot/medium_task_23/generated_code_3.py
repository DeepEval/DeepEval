import keras
from keras.layers import Input, Conv2D, Concatenate, AveragePooling2D, Flatten, Dense
from keras.layers import Reshape

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def path1(input_tensor):
        output_tensor = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return output_tensor

    def path2(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='valid', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='valid', activation='relu')(conv2)
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    def path3(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='valid', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='valid', activation='relu')(conv2)
        conv4 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='valid', activation='relu')(conv3)
        conv5 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='valid', activation='relu')(conv4)
        output_tensor = Concatenate()([conv1, conv2, conv3, conv4, conv5])
        return output_tensor

    def path4(input_tensor):
        avg_pool = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(input_tensor)
        output_tensor = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(avg_pool)
        return output_tensor

    path1_output = path1(input_tensor=input_layer)
    path2_output = path2(input_tensor=input_layer)
    path3_output = path3(input_tensor=input_layer)
    path4_output = path4(input_tensor=input_layer)

    concatenated = Concatenate()([path1_output, path2_output, path3_output, path4_output])
    output = Reshape(target_shape=(4, 4, 128))(concatenated)

    flatten = Flatten()(output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model