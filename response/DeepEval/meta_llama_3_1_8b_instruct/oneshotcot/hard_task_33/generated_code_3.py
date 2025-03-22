import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    
    def block(input_tensor):

        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)
        path3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path3)
        output_tensor = Add()([input_tensor, path1, path2, path3])

        return output_tensor
    
    branch1 = block(input_layer)
    branch2 = block(branch1)
    branch3 = block(branch2)
    concat_output = Concatenate()([branch1, branch2, branch3])
    batch_norm = BatchNormalization()(concat_output)
    flatten_layer = Flatten()(batch_norm)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model