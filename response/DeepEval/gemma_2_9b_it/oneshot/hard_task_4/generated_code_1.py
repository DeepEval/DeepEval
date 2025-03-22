import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Flatten, Dense, Reshape, multiply

def dl_model(): 
    input_layer = Input(shape=(32, 32, 3))
    
    # Increase channel dimensionality
    x = Conv2D(filters=input_layer.shape[-1] * 3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer) 

    # Depthwise separable convolution for feature extraction
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depth_multiplier=1)(x) 

    # Channel attention module
    
    # Global average pooling
    channel_pool = GlobalAveragePooling2D()(x) 

    # Two fully connected layers for attention weights
    channel_weights = Dense(units=channel_pool.shape[1], activation='sigmoid')(channel_pool) 
    channel_weights = Dense(units=channel_pool.shape[1], activation='sigmoid')(channel_weights)

    # Reshape weights to match feature map dimensions
    channel_weights = Reshape((1, 1, channel_pool.shape[1]))(channel_weights) 

    # Apply channel attention weighting
    x = multiply([x, channel_weights])

    # Reduce dimensionality with 1x1 convolution
    x = Conv2D(filters=input_layer.shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)

    # Combine with original input
    x = keras.layers.Add()([x, input_layer])

    # Flatten and fully connected layer for classification
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x) 

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model