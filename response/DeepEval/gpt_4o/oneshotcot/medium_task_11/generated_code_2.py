import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Multiply, AveragePooling2D, MaxPooling2D, Concatenate, Flatten, Activation
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Channel attention block
    # Path 1: Global Average Pooling
    gap = GlobalAveragePooling2D()(conv)
    dense1_gap = Dense(units=32, activation='relu')(gap)
    dense2_gap = Dense(units=32, activation='sigmoid')(dense1_gap)

    # Path 2: Global Max Pooling
    gmp = GlobalMaxPooling2D()(conv)
    dense1_gmp = Dense(units=32, activation='relu')(gmp)
    dense2_gmp = Dense(units=32, activation='sigmoid')(dense1_gmp)

    # Add the outputs from both paths to get channel attention weights
    channel_attention = Activation('sigmoid')(keras.layers.Add()([dense2_gap, dense2_gmp]))

    # Apply channel attention to the original features
    channel_refined_features = Multiply()([conv, channel_attention])

    # Spatial feature extraction
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(channel_refined_features)
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(channel_refined_features)

    # Concatenate spatial features
    spatial_features = Concatenate()([avg_pool, max_pool])

    # Fuse spatial features with channel features via element-wise multiplication
    fused_features = Multiply()([spatial_features, channel_refined_features])

    # Flatten and final fully connected layer
    flatten_layer = Flatten()(fused_features)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create and return the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model