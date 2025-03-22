import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    
    # 1x1 convolutional layer to increase dimensionality
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    
    # 3x3 depthwise separable convolutional layer for feature extraction
    conv2_dw = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(conv1)
    conv2_se = Conv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2_dw)
    conv2 = Add()([conv2_dw, conv2_se])
    
    # 1x1 convolutional layer to reduce dimensionality
    conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
    
    # add the original input layer to the processed output
    added_layer = Add()([conv3, input_layer])
    
    # batch normalization layer
    bath_norm = BatchNormalization()(added_layer)
    
    # flatten layer
    flatten_layer = Flatten()(bath_norm)
    
    # fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model