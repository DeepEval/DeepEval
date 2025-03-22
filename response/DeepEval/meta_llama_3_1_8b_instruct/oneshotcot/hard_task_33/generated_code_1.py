import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     
    input_layer = Input(shape=(28, 28, 1))
    
    # Define a block
    def block(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)
        path3 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path3)
        output_tensor = Concatenate()([path1, path2, path3])
        
        return output_tensor
    
    # Repeat the block three times and concatenate the outputs
    output = block(input_layer)
    output = block(output)
    output = block(output)
    output = Concatenate()([output, input_layer])
    
    # Apply batch normalization, flattening, and fully connected layer
    bath_norm = BatchNormalization()(output)
    flatten_layer = Flatten()(bath_norm)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model