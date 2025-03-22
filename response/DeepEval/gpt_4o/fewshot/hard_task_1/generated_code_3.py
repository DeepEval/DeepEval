import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Dense, Multiply, Activation, AveragePooling2D, Concatenate
from keras.layers import Add, Flatten, Reshape

def dl_model():
    input_shape = (32, 32, 3)
    num_classes = 10

    input_layer = Input(shape=input_shape)

    # Initial convolutional layer to match input channels
    initial_conv = Conv2D(filters=input_shape[-1], kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Block 1
    # Path 1: Global average pooling followed by two fully connected layers
    path1 = GlobalAveragePooling2D()(initial_conv)
    path1 = Dense(units=input_shape[-1] // 2, activation='relu')(path1)
    path1 = Dense(units=input_shape[-1], activation='relu')(path1)

    # Path 2: Global max pooling followed by two fully connected layers
    path2 = GlobalMaxPooling2D()(initial_conv)
    path2 = Dense(units=input_shape[-1] // 2, activation='relu')(path2)
    path2 = Dense(units=input_shape[-1], activation='relu')(path2)

    # Combine paths by adding and apply activation to get channel attention weights
    combined_path = Add()([path1, path2])
    channel_attention = Activation('relu')(combined_path)

    # Reshape to match input shape for multiplication
    channel_attention = Reshape((1, 1, input_shape[-1]))(channel_attention)
    channel_weighted_features = Multiply()([initial_conv, channel_attention])

    # Block 2
    # Extract spatial features
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(channel_weighted_features)
    max_pool = GlobalMaxPooling2D()(channel_weighted_features)
    max_pool = Reshape((1, 1, input_shape[-1]))(max_pool)
    spatial_features = Concatenate(axis=-1)([avg_pool, max_pool])

    # Apply 1x1 convolution and sigmoid activation
    spatial_conv = Conv2D(filters=input_shape[-1], kernel_size=(1, 1), padding='same', activation='sigmoid')(spatial_features)

    # Element-wise multiplication with channel features from Block 1
    spatial_weighted_features = Multiply()([channel_weighted_features, spatial_conv])

    # Additional branch with 1x1 convolution
    additional_branch = Conv2D(filters=input_shape[-1], kernel_size=(1, 1), padding='same')(spatial_weighted_features)

    # Combine with main path and apply activation
    combined_output = Add()([spatial_weighted_features, additional_branch])
    combined_output = Activation('relu')(combined_output)

    # Final classification layer
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=num_classes, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model