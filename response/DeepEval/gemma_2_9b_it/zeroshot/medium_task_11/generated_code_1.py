import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_tensor = keras.Input(shape=(32, 32, 3))

    # Initial convolutional layer
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)

    # Channel attention branch 1
    avg_pool = layers.GlobalAveragePooling2D()(x)
    fc1_avg = layers.Dense(128, activation='relu')(avg_pool)
    fc2_avg = layers.Dense(64, activation='relu')(fc1_avg)
    
    # Channel attention branch 2
    max_pool = layers.GlobalMaxPooling2D()(x)
    fc1_max = layers.Dense(128, activation='relu')(max_pool)
    fc2_max = layers.Dense(64, activation='relu')(fc1_max)

    # Channel attention
    channel_attention = layers.Add()([fc2_avg, fc2_max])
    channel_attention = tf.keras.layers.Activation('sigmoid')(channel_attention)
    
    # Apply channel attention weights
    x = layers.multiply([x, channel_attention])

    # Spatial feature extraction
    avg_pool_spatial = layers.AveragePooling2D((8, 8), strides=(2, 2))(x)
    max_pool_spatial = layers.MaxPooling2D((8, 8), strides=(2, 2))(x)

    # Concatenate spatial features
    spatial_features = layers.concatenate([avg_pool_spatial, max_pool_spatial], axis=3)

    # Combine channel and spatial features
    combined_features = layers.multiply([x, spatial_features])

    # Flatten and output layer
    x = layers.Flatten()(combined_features)
    output = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=input_tensor, outputs=output)
    return model