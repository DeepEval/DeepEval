import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    inputs = layers.Input(shape=(32, 32, 3))

    # Block 1: Channel Attention
    x = layers.Conv2D(3, kernel_size=3, activation='relu', padding='same')(inputs) 
    
    path1 = layers.GlobalAveragePooling2D()(x)
    path1 = layers.Dense(16, activation='relu')(path1)
    path1 = layers.Dense(3, activation='sigmoid')(path1)

    path2 = layers.GlobalMaxPooling2D()(x)
    path2 = layers.Dense(16, activation='relu')(path2)
    path2 = layers.Dense(3, activation='sigmoid')(path2)

    channel_attention = layers.Add()([path1, path2])
    channel_attention = tf.keras.layers.Multiply()([x, channel_attention])

    # Block 2: Spatial Feature Extraction
    avg_pool = layers.AveragePooling2D(pool_size=(4, 4))(channel_attention)
    max_pool = layers.MaxPooling2D(pool_size=(4, 4))(channel_attention)
    spatial_features = layers.Concatenate()([avg_pool, max_pool])
    spatial_features = layers.Conv2D(3, kernel_size=1, activation='sigmoid')(spatial_features)
    spatial_features = tf.keras.layers.Multiply()([spatial_features, channel_attention])

    # Combine Features
    x = layers.Conv2D(3, kernel_size=1, activation='relu')(spatial_features)
    x = layers.Add()([x, spatial_features])

    # Output Layer
    outputs = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model