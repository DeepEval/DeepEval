import keras
from keras.layers import Input, DepthwiseConv2D, LayerNormalization, Dense, Add

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Depthwise separable convolution layer with layer normalization
    dw_conv = DepthwiseConv2D(kernel_size=(7, 7), strides=(2, 2), depth_multiplier=1,
                              depthwise_initializer={'kernel_initializer': 'he_normal'},
                              activation='relu', use_bias=False)(input_layer)
    dw_conv = LayerNormalization(epsilon=1e-6)(dw_conv)
    
    # Two fully connected layers for channel-wise feature transformation
    dense1 = Dense(units=256, activation='relu', bias_initializer='he_normal')(dw_conv)
    dense2 = Dense(units=128, activation='relu', bias_initializer='he_normal')(dense1)
    
    # Add the original input with the processed features
    combined = Add()([dw_conv, input_layer])
    
    # Final classification layer
    output_layer = Dense(units=10, activation='softmax')(combined)
    
    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate and return the model
model = dl_model()
model.summary()