from keras.models import Model
from keras.layers import Input, DepthwiseConv2D,Dropout,Conv2D, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(28,28,1))
    deepwise_conv2d = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu6')(input_layer)
    dropout1 = Dropout(rate=0.5)(deepwise_conv2d)
    conv2d = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu6')(dropout1)
    dropout2 = Dropout(rate=0.5)(conv2d)

    flatten = Flatten()(dropout2)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
