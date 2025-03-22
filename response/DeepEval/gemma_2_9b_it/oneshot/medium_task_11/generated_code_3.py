import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Concatenate, BatchNormalization, AveragePooling2D, MaxPooling2D

def dl_model():  
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer
    conv_layer = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Parallel paths for channel feature extraction
    avg_pool_output = GlobalAveragePooling2D()(conv_layer)
    avg_pool_dense1 = Dense(units=128, activation='relu')(avg_pool_output)
    avg_pool_dense2 = Dense(units=64, activation='relu')(avg_pool_dense1)

    max_pool_output = GlobalMaxPooling2D()(conv_layer)
    max_pool_dense1 = Dense(units=128, activation='relu')(max_pool_output)
    max_pool_dense2 = Dense(units=64, activation='relu')(max_pool_dense1)

    # Channel attention mechanism
    channel_attention = Concatenate()([avg_pool_dense2, max_pool_dense2])
    channel_attention = Dense(units=32, activation='sigmoid')(channel_attention)
    channel_weighted_features = conv_layer * channel_attention

    # Spatial feature extraction
    avg_pool_spatial = AveragePooling2D(pool_size=(2, 2))(channel_weighted_features)
    max_pool_spatial = MaxPooling2D(pool_size=(2, 2))(channel_weighted_features)
    spatial_features = Concatenate()([avg_pool_spatial, max_pool_spatial])

    # Combine spatial and channel features
    fused_features = channel_weighted_features * spatial_features

    # Flatten and output layer
    flatten_layer = Flatten()(fused_features)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model