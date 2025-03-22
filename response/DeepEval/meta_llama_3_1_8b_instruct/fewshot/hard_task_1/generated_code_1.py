import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Add, Multiply, Lambda, Dense, Reshape, Activation

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer to adjust the number of output channels
    conv = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    
    def block_1(input_tensor):
        
        # Path1: Global average pooling followed by two fully connected layers
        global_avg_pool = GlobalAveragePooling2D()(input_tensor)
        flatten1 = Dense(units=128, activation='relu')(global_avg_pool)
        flatten2 = Dense(units=64, activation='relu')(flatten1)
        
        # Path2: Global max pooling followed by two fully connected layers
        global_max_pool = GlobalMaxPooling2D()(input_tensor)
        flatten3 = Dense(units=128, activation='relu')(global_max_pool)
        flatten4 = Dense(units=64, activation='relu')(flatten3)
        
        # Add the outputs of both paths
        added = Add()([flatten1, flatten2, flatten3, flatten4])
        
        # Apply channel attention weights
        channel_attention = Activation('sigmoid')(added)
        channel_attention = Reshape(target_shape=(input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]))(channel_attention)
        multiplied = Multiply()([input_tensor, channel_attention])
        
        return multiplied
    
    def block_2(input_tensor):
        
        # Extract spatial features by separately applying average pooling and max pooling
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        
        # Concatenate the outputs along the channel dimension
        concatenated = Concatenate()([avg_pool, max_pool])
        
        # Apply a 1x1 convolution and sigmoid activation to normalize the features
        normalized = Conv2D(filters=input_tensor.shape[3], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(concatenated)
        
        # Multiply the normalized features with the channel dimension features from Block 1
        spatial_attention = Multiply()([normalized, input_tensor])
        
        return spatial_attention
    
    block1_output = block_1(conv)
    block2_output = block_2(block1_output)
    
    # Ensure the output channels align with the input channels
    aligned = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(block2_output)
    
    # Add the aligned output to the main path and activate
    activated = Add()([aligned, block1_output])
    activated = Activation('relu')(activated)
    
    # Final classification through a fully connected layer
    output_layer = Dense(units=10, activation='softmax')(activated)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model