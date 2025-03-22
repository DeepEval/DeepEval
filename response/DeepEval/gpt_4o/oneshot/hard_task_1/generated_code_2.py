import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Add, Multiply, AveragePooling2D, MaxPooling2D, Concatenate, Activation
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer to adjust the number of output channels
    initial_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Block 1: Channel Attention Module
    def channel_attention_block(input_tensor):
        # Path 1: Global Average Pooling
        path1 = GlobalAveragePooling2D()(input_tensor)
        path1 = Dense(units=32, activation='relu')(path1)
        path1 = Dense(units=32, activation='sigmoid')(path1)
        
        # Path 2: Global Max Pooling
        path2 = GlobalMaxPooling2D()(input_tensor)
        path2 = Dense(units=32, activation='relu')(path2)
        path2 = Dense(units=32, activation='sigmoid')(path2)
        
        # Adding both paths
        channel_attention = Add()([path1, path2])
        
        # Multiply by input features for channel attention
        channel_attended_features = Multiply()([input_tensor, channel_attention])
        
        return channel_attended_features
    
    channel_features = channel_attention_block(initial_conv)
    
    # Block 2: Spatial Attention Module
    def spatial_attention_block(input_tensor):
        # Average Pooling
        avg_pool = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(input_tensor)
        
        # Max Pooling
        max_pool = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(input_tensor)
        
        # Concatenate along channel dimension
        concatenated_features = Concatenate(axis=-1)([avg_pool, max_pool])
        
        # 1x1 Convolution followed by sigmoid activation
        spatial_attention = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(concatenated_features)
        spatial_attention = Activation('sigmoid')(spatial_attention)
        
        # Multiply by channel attention features for spatial attention
        spatial_attended_features = Multiply()([input_tensor, spatial_attention])
        
        return spatial_attended_features
    
    spatial_features = spatial_attention_block(channel_features)
    
    # Additional branch with 1x1 convolution to align output channels with input
    branch_conv = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(spatial_features)
    
    # Add to the main path and apply activation
    added_features = Add()([input_layer, branch_conv])
    activated_features = Activation('relu')(added_features)
    
    # Final classification layer
    output_layer = Dense(units=10, activation='softmax')(activated_features)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model