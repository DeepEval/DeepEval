import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Add, Multiply, Activation, AveragePooling2D, MaxPooling2D, Concatenate, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer to adjust the number of output channels
    initial_conv = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Block 1: Channel Attention
    # Path 1
    path1_pooling = GlobalAveragePooling2D()(initial_conv)
    path1_dense1 = Dense(units=3, activation='relu')(path1_pooling)
    path1_dense2 = Dense(units=3, activation='sigmoid')(path1_dense1)
    
    # Path 2
    path2_pooling = GlobalMaxPooling2D()(initial_conv)
    path2_dense1 = Dense(units=3, activation='relu')(path2_pooling)
    path2_dense2 = Dense(units=3, activation='sigmoid')(path2_dense1)

    # Combine path1 and path2 outputs
    combined_attention = Add()([path1_dense2, path2_dense2])
    
    # Apply channel attention weights
    channel_attention = Multiply()([initial_conv, combined_attention])
    
    # Block 2: Spatial Attention
    avg_pooling = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(channel_attention)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(channel_attention)
    
    # Concatenate along the channel dimension
    spatial_features = Concatenate(axis=-1)([avg_pooling, max_pooling])
    
    # Normalize features with 1x1 convolution and sigmoid activation
    spatial_conv = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(spatial_features)
    spatial_attention = Activation('sigmoid')(spatial_conv)
    
    # Apply spatial attention
    spatial_attended_features = Multiply()([channel_attention, spatial_attention])
    
    # Additional branch with a 1x1 convolutional layer
    branch_conv = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(spatial_attended_features)
    
    # Add to the main path and activate
    final_output = Add()([spatial_attended_features, branch_conv])
    activated_output = Activation('relu')(final_output)
    
    # Final classification
    flatten = Flatten()(activated_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model