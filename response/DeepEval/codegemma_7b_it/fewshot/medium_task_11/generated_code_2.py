import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    inputs = keras.Input(shape=(32, 32, 3))

    # Initial convolutional layer
    conv = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs)

    # Feature extraction along channel dimension (Global Average Pooling and Global Max Pooling)
    gap = layers.GlobalAveragePooling2D()(conv)
    gap = layers.Dense(units=64, activation='relu')(gap)
    gap = layers.Dense(units=64, activation='relu')(gap)

    gap_output = layers.Reshape((1, 1, 64))(gap)

    gmp = layers.GlobalMaxPooling2D()(conv)
    gmp = layers.Dense(units=64, activation='relu')(gmp)
    gmp = layers.Dense(units=64, activation='relu')(gmp)

    gmp_output = layers.Reshape((1, 1, 64))(gmp)

    # Channel attention weights generation
    channel_attention = layers.Add()([gap_output, gmp_output])
    channel_attention = layers.Activation('relu')(channel_attention)
    channel_attention = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(channel_attention)
    channel_attention = layers.Activation('sigmoid')(channel_attention)

    # Applying channel attention weights to original features
    channel_weighted_features = layers.multiply([conv, channel_attention])

    # Spatial feature extraction (Average and Max Pooling)
    avg_pool = layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(channel_weighted_features)
    max_pool = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(channel_weighted_features)

    # Fusion of spatial features
    spatial_features = layers.Concatenate()([avg_pool, max_pool])

    # Combination of channel and spatial features
    concat = layers.Concatenate()([spatial_features, channel_weighted_features])
    flatten = layers.Flatten()(concat)

    # Fully connected layer and output
    dense = layers.Dense(units=64, activation='relu')(flatten)
    outputs = layers.Dense(units=10, activation='softmax')(dense)

    # Model creation
    model = Model(inputs=inputs, outputs=outputs)

    return model