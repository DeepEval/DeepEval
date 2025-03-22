import keras
from keras.layers import Input, Conv2D, AveragePooling2D, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():
    inputs = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    conv1 = Conv2D(32, (3, 3), activation='relu')(inputs)

    # Global average pooling path
    avg_pool1 = AveragePooling2D((2, 2))(conv1)
    dense1 = Dense(64, activation='relu')(avg_pool1)
    dense2 = Dense(32, activation='relu')(dense1)
    avg_pool2 = AveragePooling2D((2, 2))(dense2)

    # Global max pooling path
    max_pool1 = MaxPooling2D((2, 2))(conv1)
    dense3 = Dense(64, activation='relu')(max_pool1)
    dense4 = Dense(32, activation='relu')(dense3)
    max_pool2 = MaxPooling2D((2, 2))(dense4)

    # Channel attention weights
    channel_attention_weights = Concatenate()([avg_pool2, max_pool2])
    channel_attention_weights = Dense(32, activation='relu')(channel_attention_weights)
    channel_attention_weights = Dense(32, activation='sigmoid')(channel_attention_weights)

    # Spatial features
    spatial_features = Concatenate()([avg_pool2, max_pool2])
    spatial_features = AveragePooling2D((2, 2))(spatial_features)
    spatial_features = MaxPooling2D((2, 2))(spatial_features)

    # Fused feature map
    fused_feature_map = Concatenate()([spatial_features, channel_attention_weights])
    fused_feature_map = Flatten()(fused_feature_map)

    # Final output
    output = Dense(10, activation='softmax')(fused_feature_map)

    model = keras.Model(inputs=inputs, outputs=output)

    return model