from keras.models import Model
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add,Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(28,28,1))
    in_channel = input_layer.shape[-1]

    conv1 = Conv2D(filters=in_channel, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    deepwise_conv2d = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv2 = Conv2D(filters=in_channel, kernel_size=(1, 1), strides=(1, 1), padding='valid')(deepwise_conv2d)

    added = Add()([conv2, input_layer])
    flatten = Flatten()(added)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

