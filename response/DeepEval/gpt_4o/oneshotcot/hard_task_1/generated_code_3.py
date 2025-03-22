import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Add, Multiply
from keras.layers import AveragePooling2D, MaxPooling2D, Concatenate, Activation

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer to match input image channels
    conv_initial = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Block 1: Channel attention
    def block1(input_tensor):
        # Path 1: Global average pooling and fully connected layers
        gap = GlobalAveragePooling2D()(input_tensor)
        dense1_gap = Dense(units=32, activation='relu')(gap)
        dense2_gap = Dense(units=3, activation='sigmoid')(dense1_gap)

        # Path 2: Global max pooling and fully connected layers
        gmp = GlobalMaxPooling2D()(input_tensor)
        dense1_gmp = Dense(units=32, activation='relu')(gmp)
        dense2_gmp = Dense(units=3, activation='sigmoid')(dense1_gmp)

        # Combine both paths
        channel_attention = Add()([dense2_gap, dense2_gmp])
        channel_attention = Activation('sigmoid')(channel_attention)

        # Apply channel attention to the input
        channel_attended_features = Multiply()([input_tensor, channel_attention])

        return channel_attended_features

    channel_attended_features = block1(conv_initial)
    
    # Block 2: Spatial attention
    def block2(input_tensor):
        # Average and max pooling along the spatial dimensions
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(input_tensor)
        max_pool = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(input_tensor)

        # Concatenate the pooled features
        concat_pool = Concatenate(axis=-1)([avg_pool, max_pool])
        
        # 1x1 Convolution and sigmoid activation
        conv_spatial = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='sigmoid')(concat_pool)
        
        # Normalize and multiply with channel attended features
        spatial_attended_features = Multiply()([input_tensor, conv_spatial])
        
        return spatial_attended_features

    spatial_attended_features = block2(channel_attended_features)

    # Final output
    final_conv = Conv2D(filters=3, kernel_size=(1, 1), padding='same')(spatial_attended_features)
    final_add = Add()([final_conv, channel_attended_features])
    final_activation = Activation('relu')(final_add)

    # Final classification layer
    flatten_layer = GlobalAveragePooling2D()(final_activation)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model