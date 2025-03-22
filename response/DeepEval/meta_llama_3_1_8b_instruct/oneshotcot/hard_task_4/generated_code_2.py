import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, BatchNormalization, GlobalAveragePooling2D, Reshape, Dense, Multiply

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # 1x1 convolution to increase the dimensionality of input's channels
    conv_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # 3x3 depthwise separable convolution to extract initial features
    dw_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_1x1)
    dw_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(dw_conv)
    
    # compute channel attention weights
    avg_pool = GlobalAveragePooling2D()(dw_conv)
    weight_1 = Dense(units=64, activation='relu')(avg_pool)
    weight_1 = Dense(units=64, activation='relu')(weight_1)
    weight_1 = Reshape(target_shape=(1, 1))(weight_1)
    weight_1 = Dense(units=64, activation='sigmoid')(weight_1)
    
    # apply channel attention weighting
    attention = Multiply()([dw_conv, weight_1])
    
    # 1x1 convolution to reduce the dimensionality
    conv_1x1_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(attention)
    
    # combine the initial input and the output
    output = Concatenate()([input_layer, conv_1x1_output])
    
    # batch normalization
    bath_norm = BatchNormalization()(output)
    
    # flatten layer
    flatten_layer = Flatten()(bath_norm)
    
    # fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model