from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Multiply, Add, AveragePooling2D, MaxPooling2D, Concatenate, Activation, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.activations import sigmoid, relu, softmax

def dl_model():
    input_shape = (32, 32, 3)
    inputs = Input(shape=input_shape)

    # Initial convolutional layer to match the number of channels
    x = Conv2D(3, (3, 3), padding='same', activation='relu')(inputs)

    # Block 1 - Channel Attention
    # Path 1: Global Average Pooling + Fully Connected Layers
    gap = GlobalAveragePooling2D()(x)
    fc1 = Dense(3, activation='relu')(gap)
    fc2 = Dense(3, activation='relu')(fc1)

    # Path 2: Global Max Pooling + Fully Connected Layers
    gmp = GlobalMaxPooling2D()(x)
    fc3 = Dense(3, activation='relu')(gmp)
    fc4 = Dense(3, activation='relu')(fc3)

    # Add the outputs from Path 1 and Path 2
    added_paths = Add()([fc2, fc4])

    # Activation to generate channel attention weights
    channel_attention = Activation('relu')(added_paths)

    # Reshape to apply channel attention
    channel_attention = Dense(32 * 32 * 3, activation='relu')(channel_attention)
    channel_attention = Activation('sigmoid')(channel_attention)

    channel_attention = tf.reshape(channel_attention, (-1, 32, 32, 3))

    # Apply channel attention to original features
    channel_features = Multiply()([x, channel_attention])

    # Block 2 - Spatial Attention
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(channel_features)
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(channel_features)

    # Concatenate along channel dimension
    concatenated = Concatenate(axis=-1)([avg_pool, max_pool])

    # 1x1 Convolution and Sigmoid Activation to normalize
    spatial_attention = Conv2D(3, (1, 1), activation='sigmoid')(concatenated)

    # Multiply spatial attention with channel features
    attention_output = Multiply()([channel_features, spatial_attention])

    # Additional 1x1 Convolution to ensure channel alignment
    aligned_output = Conv2D(3, (1, 1))(attention_output)

    # Add to main path and activate
    added_output = Add()([x, aligned_output])
    activated_output = Activation('relu')(added_output)

    # Flatten and Fully Connected Layer for classification
    flattened = Flatten()(activated_output)
    output = Dense(10, activation='softmax')(flattened)

    # Build the model
    model = Model(inputs=inputs, outputs=output)

    return model