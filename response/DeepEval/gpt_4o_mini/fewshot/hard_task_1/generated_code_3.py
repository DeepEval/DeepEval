import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Add, Activation, Multiply, Concatenate, AveragePooling2D, MaxPooling2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 channels

    # Initial convolutional layer
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

    # Block 1: Channel attention
    # Path 1: Global Average Pooling
    gap = GlobalAveragePooling2D()(conv1)
    dense1 = Dense(units=32, activation='relu')(gap)
    dense2 = Dense(units=64, activation='sigmoid')(dense1)

    # Path 2: Global Max Pooling
    gmp = GlobalMaxPooling2D()(conv1)
    dense3 = Dense(units=32, activation='relu')(gmp)
    dense4 = Dense(units=64, activation='sigmoid')(dense3)

    # Channel attention weights
    channel_attention = Add()([dense2, dense4])
    channel_attention = Activation('sigmoid')(channel_attention)

    # Apply channel attention to the original features
    channel_attention_reshaped = tf.reshape(channel_attention, (-1, 1, 1, 64))
    attended_features = Multiply()([conv1, channel_attention_reshaped])

    # Block 2: Spatial feature extraction
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(attended_features)
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(attended_features)

    # Concatenate pooled features along the channel dimension
    concatenated_features = Concatenate(axis=-1)([avg_pool, max_pool])

    # Convolution followed by activation
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(concatenated_features)
    normalized_features = Activation('sigmoid')(conv2)

    # Final output features combining both blocks
    combined_features = Multiply()([attended_features, normalized_features])

    # Ensure output channels align with input channels
    output_conv = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(combined_features)

    # Adding original features to the final output path
    added_output = Add()([conv1, output_conv])
    activated_output = Activation('relu')(added_output)

    # Final classification layer
    flatten_layer = keras.layers.Flatten()(activated_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model