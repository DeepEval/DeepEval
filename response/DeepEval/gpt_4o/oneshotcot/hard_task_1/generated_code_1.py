import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Add, Multiply, AveragePooling2D, MaxPooling2D, Concatenate, Activation
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer
    conv_initial = Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    
    # Block 1: Channel Attention Mechanism
    def channel_attention_block(input_tensor):
        # Path 1: Global average pooling followed by two dense layers
        path1 = GlobalAveragePooling2D()(input_tensor)
        path1 = Dense(units=3, activation='relu')(path1)
        path1 = Dense(units=3, activation='sigmoid')(path1)
        
        # Path 2: Global max pooling followed by two dense layers
        path2 = GlobalMaxPooling2D()(input_tensor)
        path2 = Dense(units=3, activation='relu')(path2)
        path2 = Dense(units=3, activation='sigmoid')(path2)
        
        # Add the outputs of both paths and apply activation
        attention_weights = Add()([path1, path2])
        attention_weights = Activation('sigmoid')(attention_weights)
        
        # Apply channel attention weights
        channel_attention_output = Multiply()([input_tensor, attention_weights])
        
        return channel_attention_output
    
    # Block 2: Spatial Attention Mechanism
    def spatial_attention_block(input_tensor):
        # Apply average pooling and max pooling separately
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=1, padding='same')(input_tensor)
        max_pool = MaxPooling2D(pool_size=(2, 2), strides=1, padding='same')(input_tensor)
        
        # Concatenate along the channel dimension
        concat = Concatenate(axis=-1)([avg_pool, max_pool])
        
        # Apply 1x1 convolution followed by sigmoid activation
        conv = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='sigmoid')(concat)
        
        # Apply spatial attention
        spatial_attention_output = Multiply()([input_tensor, conv])
        
        return spatial_attention_output
    
    # Apply Block 1
    channel_attention_features = channel_attention_block(conv_initial)
    
    # Apply Block 2 with output from Block 1
    spatial_attention_features = spatial_attention_block(channel_attention_features)
    
    # Additional branch with 1x1 convolution
    additional_branch = Conv2D(filters=3, kernel_size=(1, 1), padding='same')(spatial_attention_features)
    
    # Add and activate the result
    output_with_attention = Add()([channel_attention_features, additional_branch])
    output_with_attention = Activation('relu')(output_with_attention)
    
    # Final classification layer
    flatten_features = GlobalAveragePooling2D()(output_with_attention)
    output_layer = Dense(units=10, activation='softmax')(flatten_features)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model