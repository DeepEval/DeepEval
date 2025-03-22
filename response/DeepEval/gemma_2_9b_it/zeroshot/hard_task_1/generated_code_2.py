import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    input_tensor = layers.Input(shape=(32, 32, 3))  

    # Block 1: Channel Attention
    x = layers.Conv2D(3, kernel_size=3, activation='relu', padding='same')(input_tensor)

    path1 = tf.keras.Sequential([
        layers.GlobalAveragePooling2D(),
        layers.Dense(32, activation='relu'),
        layers.Dense(3, activation='relu')
    ])(x)

    path2 = tf.keras.Sequential([
        layers.GlobalMaxPooling2D(),
        layers.Dense(32, activation='relu'),
        layers.Dense(3, activation='relu')
    ])(x)

    channel_attention = layers.Add()([path1, path2])
    channel_attention = layers.Activation('sigmoid')(channel_attention)
    channel_attention = tf.reshape(channel_attention, shape=(-1, 32, 32, 1))
    x = layers.multiply([x, channel_attention])

    # Block 2: Spatial Feature Extraction
    avg_pool = layers.AveragePooling2D((2, 2))(x)
    max_pool = layers.MaxPooling2D((2, 2))(x)
    spatial_features = layers.concatenate([avg_pool, max_pool], axis=3)
    spatial_features = layers.Conv2D(3, kernel_size=1, activation='sigmoid')(spatial_features)
    spatial_features = layers.multiply([x, spatial_features])

    # Final Branch
    x = layers.Conv2D(3, kernel_size=1)(spatial_features)

    # Output Layer
    output = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_tensor, outputs=output)
    return model