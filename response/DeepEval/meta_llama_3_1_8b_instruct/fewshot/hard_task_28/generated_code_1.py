import keras
from keras.layers import Input, DepthwiseConv2D, BatchNormalization, Conv2D, Add, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    def main_path(input_tensor):
        dw_conv = DepthwiseConv2D(kernel_size=(7, 7), strides=(2, 2), padding='valid', activation='relu')(input_tensor)
        bn = BatchNormalization()(dw_conv)
        pw_conv1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(bn)
        pw_conv2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pw_conv1)
        return pw_conv2
    
    def branch_path(input_tensor):
        return input_tensor
    
    main_output = main_path(input_tensor=input_layer)
    branch_output = branch_path(input_tensor=input_layer)
    adding_layer = Add()([main_output, branch_output])
    flatten_layer = Flatten()(adding_layer)
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model