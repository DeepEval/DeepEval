from keras.models import Model
from keras.layers import Input, DepthwiseConv2D, Dropout, Conv2D, Concatenate, Dense

def dl_model():

    input_layer = Input(shape=(28,28,1))

    def block(input_tensor):

        path1_depthwiseconv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu6')(input_tensor)
        path1_dropout1 = Dropout(rate=0.5)(path1_depthwiseconv)
        path1_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu6')(path1_dropout1)
        path1_dropout2 = Dropout(rate=0.5)(path1_conv)
        output_tensor = path1_dropout2

        return output_tensor

    path1 = block(input_tensor=input_layer)
    path2 = block(input_tensor=input_layer)
    path3 = block(input_tensor=input_layer)

    concatenated = Concatenate()([path1, path2, path3])

    dense1 = Dense(units=128, activation='relu')(concatenated)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

