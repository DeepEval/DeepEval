import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Add, BatchNormalization, Flatten, Dense
from keras import backend as K

def dl_model():     
    input_layer = Input(shape=(28, 28, 1))
    
    def block(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        dw_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')(input_tensor)
        pointwise_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dw_conv)
        output_tensor = Add()([conv, pointwise_conv])
        return output_tensor

    branch1 = block(input_layer)
    branch2 = block(branch1)
    branch3 = block(branch2)

    outputs = Concatenate()([branch1, branch2, branch3])
    
    batch_norm = BatchNormalization()(outputs)
    flatten_layer = Flatten()(batch_norm)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model