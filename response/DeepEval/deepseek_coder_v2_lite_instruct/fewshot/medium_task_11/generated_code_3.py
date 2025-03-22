import keras
from keras.layers import Input, Conv2D, AveragePooling2D, MaxPooling2D, Dense, Add, Activation, Multiply, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # First parallel path: Global Average Pooling followed by two fully connected layers
    gap = AveragePooling2D(pool_size=(8, 8), strides=None)(conv1)
    gap_flatten = Flatten()(gap)
    dense_gap1 = Dense(64, activation='relu')(gap_flatten)
    dense_gap2 = Dense(32, activation='relu')(dense_gap1)

    # Second parallel path: Global Max Pooling followed by two fully connected layers
    gmp = MaxPooling2D(pool_size=(8, 8), strides=None)(conv1)
    gmp_flatten = Flatten()(gmp)
    dense_gmp1 = Dense(64, activation='relu')(gmp_flatten)
    dense_gmp2 = Dense(32, activation='relu')(dense_gmp1)

    # Merge outputs from both paths
    added = Add()([dense_gap2, dense_gmp2])
    activation = Activation('relu')(added)

    # Generate channel attention weights
    channel_weights = Dense(conv1.shape[-1], activation='sigmoid')(activation)

    # Apply channel attention weights to the original features
    channel_attention = Multiply()([conv1, channel_weights])

    # Extract spatial features through average and max pooling
    avg_pool = AveragePooling2D(pool_size=(4, 4), strides=None)(channel_attention)
    max_pool = MaxPooling2D(pool_size=(4, 4), strides=None)(channel_attention)
    spatial_features = Concatenate(axis=-1)([avg_pool, max_pool])

    # Concatenate channel and spatial features
    fused_features = Multiply()([channel_attention, spatial_features])

    # Flatten and feed into a fully connected layer
    fused_flatten = Flatten()(fused_features)
    output_layer = Dense(units=10, activation='softmax')(fused_flatten)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model