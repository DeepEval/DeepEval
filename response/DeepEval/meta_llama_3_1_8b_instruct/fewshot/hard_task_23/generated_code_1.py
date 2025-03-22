import keras
from keras.layers import Input, Conv2D, AveragePooling2D, UpSampling2D, Concatenate, Conv2DTranspose, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    initial_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def branch_1(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        return conv2

    def branch_2(input_tensor):
        maxpool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool)
        upsample = UpSampling2D(size=(2, 2), interpolation='nearest')(conv)
        return upsample

    def branch_3(input_tensor):
        maxpool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool)
        upsample = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv)
        return upsample

    branch1_output = branch_1(input_tensor=initial_conv)
    branch2_output = branch_2(input_tensor=initial_conv)
    branch3_output = branch_3(input_tensor=initial_conv)

    concat_output = Concatenate()([branch1_output, branch2_output, branch3_output])
    refined_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_output)

    flatten = Flatten()(refined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model