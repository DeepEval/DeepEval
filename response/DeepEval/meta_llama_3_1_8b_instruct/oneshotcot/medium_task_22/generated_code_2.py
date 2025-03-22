import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3x3)

    def branch1(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)
        output_tensor = Concatenate()([path1, path2])

        return output_tensor

    def branch2(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)
        path3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
        output_tensor = Concatenate()([path1, path2, path3])

        return output_tensor

    def branch3(input_tensor):
        path1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        output_tensor = path1

        return output_tensor

    branch1_output = branch1(input_tensor=max_pooling)
    branch2_output = branch2(input_tensor=max_pooling)
    branch3_output = branch3(input_tensor=max_pooling)
    output_tensor = Concatenate()([branch1_output, branch2_output, branch3_output])

    bath_norm = BatchNormalization()(output_tensor)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model