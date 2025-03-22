import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Add, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    
    def branch(input_tensor):
        
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        dw_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dw_conv)
        output_tensor = Add()([conv1, conv2])
        output_tensor = Add()([output_tensor, input_tensor])
        
        return output_tensor
    
    branch1_output = branch(input_layer)
    branch2_output = branch(branch1_output)
    branch3_output = branch(branch2_output)
    
    concat_output = Concatenate()([branch1_output, branch2_output, branch3_output])
    bath_norm = BatchNormalization()(concat_output)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model