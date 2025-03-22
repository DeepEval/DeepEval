import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, UpSampling2D, Conv2DTranspose, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    def branch_1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        output_tensor = conv2
        return output_tensor
    
    def branch_2(input_tensor):
        maxpool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool)
        upsample = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(conv)
        output_tensor = upsample
        return output_tensor
    
    def branch_3(input_tensor):
        maxpool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool)
        upsample = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(conv)
        output_tensor = upsample
        return output_tensor
    
    branch1_output = branch_1(input_tensor=input_layer)
    branch2_output = branch_2(input_tensor=input_layer)
    branch3_output = branch_3(input_tensor=input_layer)
    output_tensor = Concatenate()([branch1_output, branch2_output, branch3_output])
    output_tensor = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(output_tensor)
    flatten = Flatten()(output_tensor)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model