import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Add, Activation, Concatenate, Multiply, BatchNormalization

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Initial Convolutional Layer
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

    # Block 1: Channel Attention
    # Path 1: Global Average Pooling + Fully Connected Layers
    gap_path1 = GlobalAveragePooling2D()(conv1)
    dense1_path1 = Dense(units=128, activation='relu')(gap_path1)
    dense2_path1 = Dense(units=64, activation='relu')(dense1_path1)

    # Path 2: Global Max Pooling + Fully Connected Layers
    gmp_path2 = GlobalMaxPooling2D()(conv1)
    dense1_path2 = Dense(units=128, activation='relu')(gmp_path2)
    dense2_path2 = Dense(units=64, activation='relu')(dense1_path2)

    # Adding the outputs from both paths
    channel_attention = Add()([dense2_path1, dense2_path2])
    channel_attention = Activation('sigmoid')(channel_attention)

    # Reshape to match the input's channels
    channel_attention = Dense(units=64, activation='sigmoid')(channel_attention)

    # Apply Channel Attention to the original features
    channel_attention_features = Multiply()([conv1, channel_attention])

    # Block 2: Spatial Features
    avg_pool = GlobalAveragePooling2D()(channel_attention_features)
    max_pool = GlobalMaxPooling2D()(channel_attention_features)
    
    # Concatenate the pooled features
    pooled_features = Concatenate()([avg_pool, max_pool])
    
    # 1x1 Convolution and Sigmoid Activation
    spatial_features = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='sigmoid')(channel_attention_features)

    # Normalize the features
    spatial_features = Multiply()([spatial_features, channel_attention_features])

    # Final output path
    final_output = Add()([channel_attention_features, spatial_features])
    final_output = Activation('relu')(final_output)

    # Flatten and Fully Connected Layer for classification
    flatten_layer = GlobalAveragePooling2D()(final_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Creating the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model