import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, Reshape, Multiply

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv1x1 = Conv2D(filters=3*3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1x1)
    
    def channel_attention(input_tensor):
        global_avg_pool = GlobalAveragePooling2D()(input_tensor)
        weights1 = Dense(units=32, activation='relu')(global_avg_pool)
        weights2 = Dense(units=32, activation='relu')(weights1)
        weights = Dense(units=3*3, activation='sigmoid')(weights2)
        weights = Reshape((3*3,))(weights)
        weights = Multiply()([weights, input_tensor])
        return weights
    
    weights = channel_attention(depthwise_conv)
    combined = Concatenate()([depthwise_conv, weights])
    reduced = Conv2D(filters=3*3, kernel_size=(1, 1), strides=(1, 1), padding='same')(combined)
    
    output_layer = Subtract()([input_layer, reduced])  # Subtract the reduced features from the input to achieve the described effect
    
    bath_norm = BatchNormalization()(output_layer)
    flatten_layer = Flatten()(bath_norm)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model