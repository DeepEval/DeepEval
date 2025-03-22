import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Add, AveragePooling2D, MaxPooling2D, Concatenate

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    conv_layer = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Channel Attention Path 1: Global Average Pooling
    avg_pool_layer = GlobalAveragePooling2D()(conv_layer)
    fc1_1 = Dense(units=128, activation='relu')(avg_pool_layer)
    fc2_1 = Dense(units=64, activation='relu')(fc1_1)
    
    # Channel Attention Path 2: Global Max Pooling
    max_pool_layer = GlobalMaxPooling2D()(conv_layer)
    fc1_2 = Dense(units=128, activation='relu')(max_pool_layer)
    fc2_2 = Dense(units=64, activation='relu')(fc1_2)
    
    # Combine channel attention outputs
    channel_attention = Add()([fc2_1, fc2_2])
    channel_attention = keras.layers.Activation('sigmoid')(channel_attention)
    
    # Apply channel attention weights
    weighted_features = conv_layer * channel_attention

    # Spatial Feature Extraction
    avg_pool_spatial = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(weighted_features)
    max_pool_spatial = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(weighted_features)
    
    # Concatenate spatial features
    fused_features = Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])

    # Combine channel and spatial features
    combined_features = weighted_features * fused_features

    # Final Classification
    flatten_layer = Flatten()(combined_features)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model