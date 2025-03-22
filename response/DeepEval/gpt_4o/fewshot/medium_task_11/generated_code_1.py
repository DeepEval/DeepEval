import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Multiply, AveragePooling2D, MaxPooling2D, Concatenate, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial Convolutional Layer
    initial_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Channel Dimension Feature Extraction
    # Path 1: Global Average Pooling followed by two Dense layers
    gap = GlobalAveragePooling2D()(initial_conv)
    fc1_gap = Dense(units=64, activation='relu')(gap)
    fc2_gap = Dense(units=64, activation='relu')(fc1_gap)

    # Path 2: Global Max Pooling followed by two Dense layers
    gmp = GlobalMaxPooling2D()(initial_conv)
    fc1_gmp = Dense(units=64, activation='relu')(gmp)
    fc2_gmp = Dense(units=64, activation='relu')(fc1_gmp)

    # Adding outputs of both paths
    added_features = keras.layers.Add()([fc2_gap, fc2_gmp])
    channel_attention_weights = Dense(units=64, activation='sigmoid')(added_features)

    # Apply channel attention weights
    channel_attention = Multiply()([initial_conv, channel_attention_weights])

    # Spatial Feature Extraction
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(channel_attention)
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(channel_attention)
    spatial_features = Concatenate(axis=-1)([avg_pool, max_pool])

    # Combine Spatial and Channel Features
    combined_features = Multiply()([channel_attention, spatial_features])
    flatten = Flatten()(combined_features)

    # Final Fully Connected Layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model