import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Multiply, Flatten, Add

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Step 1: Increase dimensionality of the input's channels threefold using 1x1 convolution
    expanded_channels = Conv2D(filters=9, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 2: Extract initial features using a 3x3 depthwise separable convolution
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(expanded_channels)
    
    # Step 3: Compute channel attention weights
    # Global Average Pooling
    global_avg_pool = GlobalAveragePooling2D()(depthwise_conv)
    
    # Fully connected layers for attention mechanism
    dense1 = Dense(units=16, activation='relu')(global_avg_pool)
    attention_weights = Dense(units=9, activation='sigmoid')(dense1)
    
    # Reshape and apply channel attention weights
    attention_weights_reshaped = keras.layers.Reshape((1, 1, 9))(attention_weights)
    channel_attention_weighted = Multiply()([depthwise_conv, attention_weights_reshaped])
    
    # Step 4: Reduce dimensionality with a 1x1 convolution
    reduced_channels = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(channel_attention_weighted)
    
    # Step 5: Combine with initial input
    combined = Add()([reduced_channels, input_layer])
    
    # Step 6: Pass through a flattening layer
    flatten_layer = Flatten()(combined)
    
    # Step 7: Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model