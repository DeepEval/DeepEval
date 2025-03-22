import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, Dropout, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))
    
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    dropout1 = Dropout(0.25)(depthwise_conv)
    
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout1)
    dropout2 = Dropout(0.25)(conv1)
    
    flatten_layer = Flatten()(dropout2)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model