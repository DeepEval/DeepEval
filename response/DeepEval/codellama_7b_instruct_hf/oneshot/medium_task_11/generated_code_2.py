import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Add, Activation, Multiply, Permute

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Global average pooling
    avg_pool = GlobalAveragePooling2D()(conv)
    avg_pool_flat = Flatten()(avg_pool)
    avg_fc1 = Dense(units=64, activation='relu')(avg_pool_flat)
    avg_fc2 = Dense(units=32, activation='relu')(avg_fc1)

    # Global max pooling
    max_pool = GlobalMaxPooling2D()(conv)
    max_pool_flat = Flatten()(max_pool)
    max_fc1 = Dense(units=64, activation='relu')(max_pool_flat)
    max_fc2 = Dense(units=32, activation='relu')(max_fc1)

    # Channel attention weights
    channel_attention = Add()([avg_fc2, max_fc2])
    channel_attention = Activation('sigmoid')(channel_attention)
    channel_attention = Multiply()([channel_attention, conv])

    # Spatial features
    spatial_features = Concatenate()([
        MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv),
        MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(conv)
    ])
    spatial_features = Flatten()(spatial_features)
    spatial_features = Dense(units=64, activation='relu')(spatial_features)
    spatial_features = Dense(units=32, activation='relu')(spatial_features)

    # Fused feature map
    fused_feature_map = Concatenate()([channel_attention, spatial_features])
    fused_feature_map = Flatten()(fused_feature_map)
    fused_feature_map = Dense(units=128, activation='relu')(fused_feature_map)
    fused_feature_map = Dense(units=64, activation='relu')(fused_feature_map)

    # Final output
    output_layer = Dense(units=10, activation='softmax')(fused_feature_map)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model