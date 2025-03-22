import keras
from keras.layers import (Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, 
                          Dense, Add, Activation, AveragePooling2D, MaxPooling2D, 
                          Concatenate, Multiply, Reshape, Flatten)

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer to match input channels
    initial_conv = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)

    # Block 1: Channel Attention
    def channel_attention(input_tensor):
        # Path 1
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        dense1_avg = Dense(units=3, activation='relu')(avg_pool)
        dense2_avg = Dense(units=3, activation='relu')(dense1_avg)
        
        # Path 2
        max_pool = GlobalMaxPooling2D()(input_tensor)
        dense1_max = Dense(units=3, activation='relu')(max_pool)
        dense2_max = Dense(units=3, activation='relu')(dense1_max)
        
        # Combine paths
        combined = Add()([dense2_avg, dense2_max])
        channel_attention_weights = Activation('sigmoid')(combined)
        channel_attention_weights = Reshape((1, 1, 3))(channel_attention_weights)
        channel_refined = Multiply()([input_tensor, channel_attention_weights])
        
        return channel_refined

    # Block 2: Spatial Attention
    def spatial_attention(input_tensor):
        # Average and Max pooling
        avg_pool_spatial = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(input_tensor)
        max_pool_spatial = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(input_tensor)
        
        # Concatenate
        concatenated = Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])
        
        # 1x1 Convolution and sigmoid activation
        spatial_attention_weights = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='sigmoid')(concatenated)
        spatial_refined = Multiply()([input_tensor, spatial_attention_weights])
        
        return spatial_refined

    # Apply channel attention
    channel_attention_output = channel_attention(initial_conv)
    
    # Apply spatial attention
    spatial_attention_output = spatial_attention(channel_attention_output)

    # Additional branch with 1x1 convolution
    additional_conv = Conv2D(filters=3, kernel_size=(1, 1), padding='same')(spatial_attention_output)
    output_main_path = Add()([initial_conv, additional_conv])
    activated_output = Activation('relu')(output_main_path)
    
    # Final classification layer
    flat_output = Flatten()(activated_output)
    final_output = Dense(units=10, activation='softmax')(flat_output)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=final_output)
    
    return model

# Instantiate and inspect the model
model = dl_model()
model.summary()