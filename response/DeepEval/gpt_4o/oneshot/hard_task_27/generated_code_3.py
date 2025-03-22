import keras
from keras.layers import Input, DepthwiseConv2D, LayerNormalization, Dense, Add, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Depthwise Separable Convolution with 7x7 kernel
    depthwise_conv = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Layer Normalization
    layer_norm = LayerNormalization()(depthwise_conv)
    
    # Channel-wise feature transformation with fully connected layers
    flatten_layer = Flatten()(layer_norm)
    dense1 = Dense(units=32 * 32 * 3, activation='relu')(flatten_layer) # Same number of channels as input
    dense2 = Dense(units=32 * 32 * 3, activation='relu')(dense1) # Same number of channels as input
    
    # Reshape to match input dimensions for addition
    reshaped_dense = keras.layers.Reshape((32, 32, 3))(dense2)
    
    # Combine original input with processed features
    combined_output = Add()([input_layer, reshaped_dense])
    
    # Final classification layers
    flatten_combined = Flatten()(combined_output)
    final_dense1 = Dense(units=128, activation='relu')(flatten_combined)
    final_dense2 = Dense(units=64, activation='relu')(final_dense1)
    output_layer = Dense(units=10, activation='softmax')(final_dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model