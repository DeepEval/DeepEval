import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Concatenate, Add, AveragePooling2D, MaxPooling2D

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Channel Attention Branch 1: Global Average Pooling
    avg_pool = GlobalAveragePooling2D()(x)
    fc1_avg = Dense(units=128, activation='relu')(avg_pool)
    fc2_avg = Dense(units=64, activation='relu')(fc1_avg)
    avg_attn = Dense(units=32, activation='sigmoid')(fc2_avg) 

    # Channel Attention Branch 2: Global Max Pooling
    max_pool = GlobalMaxPooling2D()(x)
    fc1_max = Dense(units=128, activation='relu')(max_pool)
    fc2_max = Dense(units=64, activation='relu')(fc1_max)
    max_attn = Dense(units=32, activation='sigmoid')(fc2_max)

    # Add attention outputs and apply to input
    channel_attn = Add()([avg_attn, max_attn])
    x = x * channel_attn

    # Spatial Feature Extraction
    avg_pool_spatial = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    max_pool_spatial = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    spatial_features = Concatenate(axis=3)([avg_pool_spatial, max_pool_spatial])

    # Combine Channel and Spatial Features
    combined_features = x * spatial_features

    # Flatten and Output Layer
    flatten = Flatten()(combined_features)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model