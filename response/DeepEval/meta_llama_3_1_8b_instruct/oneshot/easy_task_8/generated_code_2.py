import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    dropout1 = Dropout(0.2)(depthwise_conv)
    
    conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout1)
    dropout2 = Dropout(0.2)(conv)
    
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(dropout2)
    
    flatten_layer = Flatten()(max_pooling)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense_layer)

    return model