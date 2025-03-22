import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    init_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def branch1(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        return conv2

    def branch2(input_tensor):
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool)
        upsample = UpSampling2D(size=(2, 2))(conv)
        return upsample

    def branch3(input_tensor):
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool)
        upsample = UpSampling2D(size=(2, 2))(conv)
        return upsample

    path1 = branch1(init_conv)
    path2 = branch2(init_conv)
    path3 = branch3(init_conv)
    concat_output = Concatenate()([path1, path2, path3])
    conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_output)
    bath_norm = BatchNormalization()(conv)
    flatten_layer = Flatten()(bath_norm)
    dense_layer = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model