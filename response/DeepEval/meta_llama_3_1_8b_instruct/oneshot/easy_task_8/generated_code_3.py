import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, Dropout, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    depthwise_conv = Dropout(0.2)(depthwise_conv)  # Add dropout to mitigate overfitting
    
    conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)
    conv = Dropout(0.2)(conv)  # Add dropout to mitigate overfitting
    
    flatten_layer = Flatten()(conv)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense_layer)

    return model