from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Activation, Dense, Flatten, concatenate, multiply
from tensorflow.keras import Model

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    conv1 = Conv2D(filters=16, kernel_size=3, padding='same')(inputs)
    conv1 = Activation('relu')(conv1)

    # Feature extraction along the channel dimension
    # Path 1: Global Average Pooling
    avg_pool = GlobalAveragePooling2D()(conv1)
    fc1 = Dense(16, activation='relu')(avg_pool)
    fc2 = Dense(16, activation='sigmoid')(fc1)

    # Path 2: Global Max Pooling
    max_pool = GlobalMaxPooling2D()(conv1)
    fc3 = Dense(16, activation='relu')(max_pool)
    fc4 = Dense(16, activation='sigmoid')(fc3)

    # Channel attention weights
    channel_attention = multiply([avg_pool, max_pool])
    channel_attention = Activation('sigmoid')(channel_attention)

    # Apply channel attention weights to original features
    channel_gated = multiply([conv1, channel_attention])

    # Spatial feature extraction
    # Average pooling
    avg_pool_spatial = GlobalAveragePooling2D()(channel_gated)
    fc5 = Dense(16, activation='relu')(avg_pool_spatial)

    # Max pooling
    max_pool_spatial = GlobalMaxPooling2D()(channel_gated)
    fc6 = Dense(16, activation='relu')(max_pool_spatial)

    # Fused feature map
    spatial_features = concatenate([avg_pool_spatial, max_pool_spatial])
    spatial_features = Dense(16, activation='relu')(spatial_features)

    # Combine channel and spatial features
    combined_features = multiply([spatial_features, channel_gated])

    # Output layer
    flatten = Flatten()(combined_features)
    outputs = Dense(10, activation='softmax')(flatten)

    # Model construction
    model = Model(inputs=inputs, outputs=outputs)

    return model