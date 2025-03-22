import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Multiply, AveragePooling2D, MaxPooling2D, Concatenate, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial Convolutional Layer
    conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Channel Attention
    # Path 1: Global Average Pooling
    gap = GlobalAveragePooling2D()(conv)
    fc1_gap = Dense(units=32, activation='relu')(gap)
    fc2_gap = Dense(units=64, activation='sigmoid')(fc1_gap)
    
    # Path 2: Global Max Pooling
    gmp = GlobalMaxPooling2D()(conv)
    fc1_gmp = Dense(units=32, activation='relu')(gmp)
    fc2_gmp = Dense(units=64, activation='sigmoid')(fc1_gmp)
    
    # Combine Paths and Apply Channel Attention
    channel_attention = Multiply()([conv, fc2_gap, fc2_gmp])
    
    # Spatial Feature Extraction
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(channel_attention)
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(channel_attention)
    
    # Concatenate Spatial Features
    spatial_features = Concatenate(axis=-1)([avg_pool, max_pool])
    
    # Combine Spatial and Channel Features
    combined_features = Multiply()([channel_attention, spatial_features])
    
    # Flatten and Fully Connected Layer for Classification
    flatten_layer = Flatten()(combined_features)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model