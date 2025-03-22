import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32x3

    def multi_scale_block(input_tensor):
        # Path 1: Single 1x1 convolution
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Path 2: Average pooling followed by 1x1 convolution
        path2 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)
        
        # Path 3: 1x1 convolution followed by two parallel 1x3 and 3x1 convolutions
        path3_initial = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3_1 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path3_initial)
        path3_2 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path3_initial)
        path3 = Concatenate()([path3_1, path3_2])
        
        # Path 4: 1x1 convolution followed by 3x3 convolution, then two parallel 1x3 and 3x1 convolutions
        path4_initial = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path4_initial)
        path4_1 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path4_3x3)
        path4_2 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path4_3x3)
        path4 = Concatenate()([path4_1, path4_2])
        
        # Concatenate all paths
        output_tensor = Concatenate()([path1, path2, path3, path4])
        
        return output_tensor
    
    block_output = multi_scale_block(input_layer)
    flatten_layer = Flatten()(block_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model