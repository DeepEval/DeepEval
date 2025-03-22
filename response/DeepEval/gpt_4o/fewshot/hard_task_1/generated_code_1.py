import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Add, Multiply, AveragePooling2D, MaxPooling2D, Concatenate, Activation, Flatten
from keras.activations import sigmoid, relu

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images have 32x32 size with 3 channels

    # Initial convolutional layer to adjust the number of channels
    initial_conv = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Block 1
    # Path1: Global average pooling followed by two fully connected layers
    gap = GlobalAveragePooling2D()(initial_conv)
    fc1_path1 = Dense(units=3, activation='relu')(gap)
    fc2_path1 = Dense(units=3, activation='relu')(fc1_path1)

    # Path2: Global max pooling followed by two fully connected layers
    gmp = GlobalMaxPooling2D()(initial_conv)
    fc1_path2 = Dense(units=3, activation='relu')(gmp)
    fc2_path2 = Dense(units=3, activation='relu')(fc1_path2)

    # Adding the outputs from both paths
    added_paths = Add()([fc2_path1, fc2_path2])

    # Activation to generate channel attention weights
    channel_attention_weights = Activation('sigmoid')(added_paths)

    # Applying the attention weights to the original features
    channel_scaled_features = Multiply()([initial_conv, channel_attention_weights])

    # Block 2
    # Extract spatial features by separately applying average pooling and max pooling
    avg_pooled = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(channel_scaled_features)
    max_pooled = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(channel_scaled_features)

    # Concatenate along the channel dimension
    concatenated_spatial_features = Concatenate(axis=-1)([avg_pooled, max_pooled])

    # 1x1 convolution and sigmoid activation to normalize features
    spatial_features = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(concatenated_spatial_features)
    normalized_spatial_features = Activation('sigmoid')(spatial_features)

    # Multiply element-wise with the channel dimension features
    combined_features = Multiply()([channel_scaled_features, normalized_spatial_features])

    # Additional branch with a 1x1 convolutional layer
    additional_branch = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(combined_features)

    # Add the result to the main path and activate
    final_features = Add()([combined_features, additional_branch])
    activated_features = Activation('relu')(final_features)

    # Final classification through a fully connected layer
    flatten_layer = Flatten()(activated_features)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model