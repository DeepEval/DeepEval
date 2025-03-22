import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    def branch1(input_tensor):
        path1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return path1

    def branch2(input_tensor):
        path2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
        return path3

    def branch3(input_tensor):
        path3 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(path3)
        return path4

    def branch4(input_tensor):
        path4 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(input_tensor)
        path5 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)
        return path5

    path1 = branch1(conv)
    path2 = branch2(conv)
    path3 = branch3(conv)
    path4 = branch4(conv)
    output_tensor = Concatenate()([path1, path2, path3, path4])

    bath_norm = BatchNormalization()(output_tensor)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model