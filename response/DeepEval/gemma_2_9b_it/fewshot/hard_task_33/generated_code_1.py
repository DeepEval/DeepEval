import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Flatten, Dense

def dl_model():  
    
    input_layer = Input(shape=(28,28,1))
    
    def block(input_tensor):
        x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
        output_tensor = Add()([x, input_tensor])
        return output_tensor

    branch1 = block(input_layer)
    branch2 = block(input_layer)
    branch3 = block(input_layer)
    
    merged_tensor = Concatenate()([branch1, branch2, branch3])
    flatten_layer = Flatten()(merged_tensor)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model