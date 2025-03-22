import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Concatenate, multiply, Reshape, Lambda

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Increase the dimensionality of the input channels
    conv1 = Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Depthwise separable convolution
    dw_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(conv1)
    
    # Global average pooling
    avg_pool = GlobalAveragePooling2D()(dw_conv)
    
    # Two fully connected layers for channel attention
    fc1 = Dense(units=512, activation='relu')(avg_pool)
    fc2 = Dense(units=256, activation='relu')(fc1)
    
    # Generate weights for channel attention
    attention_weights = Dense(units=dw_conv.shape[1], activation='softmax')(fc2)
    
    # Reshape attention weights
    reshaped_attention_weights = Reshape((dw_conv.shape[1],))(attention_weights)
    
    # Channel attention
    weighted_features = multiply([dw_conv, reshaped_attention_weights])
    
    # Reduce dimensionality
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(weighted_features)
    
    # Combine with initial input
    combined_output = Concatenate()([conv1, conv2])
    
    # Flatten and fully connected layers
    flatten = Flatten()(combined_output)
    dense1 = Dense(units=512, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=dense2)
    
    return model