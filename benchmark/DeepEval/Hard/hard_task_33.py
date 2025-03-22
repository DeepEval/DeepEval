from keras.models import Model
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add,Concatenate, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(28,28,1))
    
    def block(input_tensor):

        input_channel = input_tensor.shape[-1]
        path1_conv = Conv2D(filters=input_channel, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path1_depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1_conv)
        path1_conv = Conv2D(filters=input_channel, kernel_size=(1, 1), strides=(1, 1), padding='valid')(path1_depthwise_conv)
        output_tensor = Add()([path1_conv, input_tensor])

        return output_tensor
    
    path1 = block(input_tensor=input_layer)
    path2 = block(input_tensor=input_layer)
    path3 = block(input_tensor=input_layer)
    concatenated = Concatenate()([path1, path2, path3])

    flatten = Flatten()(concatenated)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    model = Model(inputs=input_layer, outputs=output_layer)

    return model


