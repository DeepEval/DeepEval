import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Add, Activation, Multiply, Concatenate, BatchNormalization

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Initial convolutional layer
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Block 1: Channel attention mechanism
    # Path 1: Global Average Pooling followed by two fully connected layers
    gap_path = GlobalAveragePooling2D()(conv1)
    dense1_path1 = Dense(units=128, activation='relu')(gap_path)
    dense2_path1 = Dense(units=64, activation='relu')(dense1_path1)

    # Path 2: Global Max Pooling followed by two fully connected layers
    gmp_path = GlobalMaxPooling2D()(conv1)
    dense1_path2 = Dense(units=128, activation='relu')(gmp_path)
    dense2_path2 = Dense(units=64, activation='relu')(dense1_path2)

    # Combine both paths
    combined_path = Add()([dense2_path1, dense2_path2])
    channel_attention_weights = Activation('sigmoid')(combined_path)

    # Apply channel attention to the original features
    channel_attention = Multiply()([conv1, channel_attention_weights])

    # Block 2: Spatial feature extraction
    avg_pool = GlobalAveragePooling2D()(channel_attention)
    max_pool = GlobalMaxPooling2D()(channel_attention)
    spatial_features = Concatenate()([avg_pool, max_pool])

    # Apply 1x1 convolution and sigmoid activation
    spatial_conv = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(spatial_features)
    normalized_features = Activation('sigmoid')(spatial_conv)

    # Element-wise multiplication with the output from Block 1
    refined_features = Multiply()([channel_attention, normalized_features])

    # Additional branch with a 1x1 convolutional layer for output channel alignment
    additional_branch = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(refined_features)

    # Add to the main path and activate
    final_output = Add()([channel_attention, additional_branch])
    final_output = Activation('relu')(final_output)

    # Final fully connected layer for classification
    flatten_layer = GlobalAveragePooling2D()(final_output)  # Flattening for dense layer
    classification_output = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=classification_output)

    return model