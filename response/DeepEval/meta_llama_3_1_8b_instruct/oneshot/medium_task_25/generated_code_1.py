import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Path 1: single 1x1 convolution
    path1_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Path 2: average pooling followed by a 1x1 convolution
    path2_avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    path2_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2_avg_pool)
    
    # Path 3: 1x1 convolution followed by two parallel 1x3 and 3x1 convolutions
    path3_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3_conv_1x3 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path3_conv)
    path3_conv_3x1 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path3_conv)
    path3_concat = Concatenate()([path3_conv_1x3, path3_conv_3x1])
    
    # Path 4: 1x1 convolution followed by a 3x3 convolution, then followed by two parallel 1x3 and 3x1 convolutions
    path4_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path4_conv_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path4_conv)
    path4_conv_3x3_1x3 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path4_conv_3x3)
    path4_conv_3x3_3x1 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path4_conv_3x3)
    path4_concat = Concatenate()([path4_conv_3x3_1x3, path4_conv_3x3_3x1])
    
    # multi-scale feature fusion
    multi_scale_feature = Concatenate()([path1_conv, path2_conv, path3_concat, path4_concat])
    bath_norm = BatchNormalization()(multi_scale_feature)
    flatten_layer = Flatten()(bath_norm)
    
    # fully connected layer for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model