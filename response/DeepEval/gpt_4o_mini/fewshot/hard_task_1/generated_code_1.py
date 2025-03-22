import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Add, Activation, Multiply, Concatenate, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels (RGB)

    # Initial convolutional layer
    initial_conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

    # Block 1
    # Path 1: Global Average Pooling + Fully Connected Layers
    avg_pool = GlobalAveragePooling2D()(initial_conv)
    dense1_avg = Dense(units=128, activation='relu')(avg_pool)
    dense2_avg = Dense(units=64, activation='relu')(dense1_avg)

    # Path 2: Global Max Pooling + Fully Connected Layers
    max_pool = GlobalMaxPooling2D()(initial_conv)
    dense1_max = Dense(units=128, activation='relu')(max_pool)
    dense2_max = Dense(units=64, activation='relu')(dense1_max)

    # Channel attention weights
    channel_attention = Add()([dense2_avg, dense2_max])
    channel_attention = Activation('sigmoid')(channel_attention)
    channel_attention = tf.expand_dims(channel_attention, axis=1)  # Make it compatible for multiplication

    # Apply channel attention to the original features
    channel_attention_features = Multiply()([initial_conv, channel_attention])

    # Block 2: Extract spatial features
    avg_pool_2 = GlobalAveragePooling2D()(channel_attention_features)
    max_pool_2 = GlobalMaxPooling2D()(channel_attention_features)
    
    # Concatenate the outputs along the channel dimension
    spatial_features = Concatenate()([avg_pool_2, max_pool_2])
    
    # 1x1 Convolution and normalization
    spatial_features = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='sigmoid')(tf.reshape(spatial_features, (-1, 1, 1, 128)))

    # Multiply spatial features with channel attention features
    combined_features = Multiply()([channel_attention_features, spatial_features])

    # Additional branch with 1x1 convolution
    additional_branch = Conv2D(filters=3, kernel_size=(1, 1), padding='same')(combined_features)

    # Add to the main path
    added_features = Add()([combined_features, additional_branch])
    activated_output = Activation('relu')(added_features)

    # Final classification layer
    flatten = Flatten()(activated_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)  # CIFAR-10 has 10 classes

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model