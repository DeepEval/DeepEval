import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Concatenate, Add, Flatten

def dl_model():
    input_tensor = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)

    # Channel attention branch 1: Global Average Pooling
    avg_pool = GlobalAveragePooling2D()(x)
    avg_fc1 = Dense(128, activation='relu')(avg_pool)
    avg_fc2 = Dense(64, activation='relu')(avg_fc1)

    # Channel attention branch 2: Global Max Pooling
    max_pool = GlobalMaxPooling2D()(x)
    max_fc1 = Dense(128, activation='relu')(max_pool)
    max_fc2 = Dense(64, activation='relu')(max_fc1)

    # Combine channel attention outputs
    channel_attention = Add()([avg_fc2, max_fc2])
    channel_attention = tf.nn.sigmoid(channel_attention)

    # Apply channel attention weights
    attended_features = x * channel_attention

    # Spatial feature extraction
    avg_pool_spatial = tf.keras.layers.AveragePooling2D((8, 8))(attended_features)
    max_pool_spatial = tf.keras.layers.MaxPooling2D((8, 8))(attended_features)
    spatial_features = Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])

    # Combine channel and spatial features
    fused_features = attended_features * spatial_features

    # Flatten and output layer
    x = Flatten()(fused_features)
    output = Dense(10, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=output)

    return model