from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Multiply, Add, Flatten, AveragePooling2D, Concatenate, Activation
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))  # CIFAR-10 images have shape 32x32x3

    # Initial convolutional layer
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)

    # Channel attention mechanism
    # Parallel path 1: Global Average Pooling
    gap = GlobalAveragePooling2D()(x)
    fc1 = Dense(32, activation='relu')(gap)
    fc1 = Dense(32, activation='relu')(fc1)

    # Parallel path 2: Global Max Pooling
    gmp = GlobalMaxPooling2D()(x)
    fc2 = Dense(32, activation='relu')(gmp)
    fc2 = Dense(32, activation='relu')(fc2)

    # Add outputs from the two paths
    channel_attention = Add()([fc1, fc2])
    channel_attention = Activation('sigmoid')(channel_attention)

    # Apply channel attention to original features
    channel_attention = Multiply()([x, channel_attention])

    # Spatial feature extraction
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(channel_attention)
    max_pool = GlobalMaxPooling2D()(channel_attention)
    max_pool = tf.keras.layers.Reshape((1, 1, 32))(max_pool)  # Reshape to match spatial dimensions
    spatial_features = Concatenate(axis=-1)([avg_pool, max_pool])

    # Combine spatial and channel features
    combined_features = Multiply()([channel_attention, spatial_features])

    # Flatten and final fully connected layer
    flat = Flatten()(combined_features)
    outputs = Dense(10, activation='softmax')(flat)  # CIFAR-10 has 10 classes

    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)

    return model