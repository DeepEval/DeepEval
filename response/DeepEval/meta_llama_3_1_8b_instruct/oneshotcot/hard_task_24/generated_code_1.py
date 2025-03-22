import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, UpSampling2D, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    initial_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def branch1(input_tensor):
        return Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)

    def branch2(input_tensor):
        downsample = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_tensor)
        conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(downsample)
        upsample = UpSampling2D(size=(2, 2))(conv)
        return upsample

    def branch3(input_tensor):
        downsample = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_tensor)
        conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(downsample)
        upsample = UpSampling2D(size=(2, 2))(conv)
        return upsample

    branch1_output = branch1(initial_conv)
    branch2_output = branch2(initial_conv)
    branch3_output = branch3(initial_conv)
    concat_output = Concatenate()([branch1_output, branch2_output, branch3_output])
    final_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_output)

    bath_norm = BatchNormalization()(final_conv)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model