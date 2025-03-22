import keras
from keras.layers import Input, Conv2D, LayerNormalization, Dense, Add

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3)) 

    # 7x7 depthwise separable convolution with layer normalization
    conv_layer = Conv2D(filters=128, kernel_size=(7, 7), strides=(2, 2), 
                        depth_multiplier=1, padding='same', activation='relu')(input_layer)
    norm_layer = LayerNormalization()(conv_layer)

    # Two fully connected layers for channel-wise feature transformation
    flatten_layer = Flatten()(norm_layer)
    dense1 = Dense(units=3*32*32, activation='relu')(flatten_layer) 
    dense2 = Dense(units=3*32*32, activation='relu')(dense1)
    
    # Combine original input and processed features
    combined_features = Add()([input_layer, dense2])

    # Final classification layers
    output_layer = Dense(units=10, activation='softmax')(combined_features)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model