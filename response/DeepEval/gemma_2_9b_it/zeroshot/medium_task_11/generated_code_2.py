import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    inputs = layers.Input(shape=(32, 32, 3))

    # Initial convolutional layer
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)

    # Parallel paths with global average and max pooling
    avg_pool = layers.GlobalAveragePooling2D()(x)
    avg_dense1 = layers.Dense(128, activation='relu')(avg_pool)
    avg_dense2 = layers.Dense(64, activation='relu')(avg_dense1)

    max_pool = layers.GlobalMaxPooling2D()(x)
    max_dense1 = layers.Dense(128, activation='relu')(max_pool)
    max_dense2 = layers.Dense(64, activation='relu')(max_dense1)

    # Channel attention
    attention_weights = layers.Add()([avg_dense2, max_dense2])
    attention_weights = layers.Activation('sigmoid')(attention_weights)

    x = layers.multiply([x, attention_weights])

    # Spatial feature extraction
    avg_pool = layers.AveragePooling2D((2, 2))(x)
    max_pool = layers.MaxPooling2D((2, 2))(x)
    spatial_features = layers.concatenate([avg_pool, max_pool], axis=3)

    # Fusion of channel and spatial features
    fused_features = layers.multiply([x, spatial_features])

    # Flatten and output layer
    x = layers.Flatten()(fused_features)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model