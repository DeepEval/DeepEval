import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))

    def block_1(input_tensor):
        relu = keras.layers.ReLU()(input_tensor)
        conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(relu)
        maxpool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv)
        return maxpool
    
    def block_2(input_tensor):
        relu = keras.layers.ReLU()(input_tensor)
        conv = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same')(relu)
        maxpool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv)
        return maxpool
    
    main_path = block_2(block_1(input_layer))
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    adding_layer = Add()([main_path, branch_path])
    
    flatten_layer = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model