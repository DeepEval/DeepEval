from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Dense, Add, Multiply, Activation, AveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Concatenate, Reshape
from tensorflow.keras.models import Model

def dl_model():
    input_shape = (32, 32, 3)
    inputs = Input(shape=input_shape)
    
    # Initial convolutional layer
    x = Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    
    # Block 1 - Channel Attention Block
    # Path 1
    path1 = GlobalAveragePooling2D()(x)
    path1 = Dense(3, activation='relu')(path1)
    path1 = Dense(3, activation='relu')(path1)
    
    # Path 2
    path2 = GlobalMaxPooling2D()(x)
    path2 = Dense(3, activation='relu')(path2)
    path2 = Dense(3, activation='relu')(path2)
    
    # Adding both paths
    channel_attention = Add()([path1, path2])
    channel_attention = Activation('sigmoid')(channel_attention)
    
    # Reshape channel attention to match input shape
    channel_attention = Reshape((1, 1, 3))(channel_attention)
    channel_features = Multiply()([x, channel_attention])
    
    # Block 2 - Spatial Attention Block
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(channel_features)
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(channel_features)
    
    # Concatenate along the channel dimension
    spatial_features = Concatenate(axis=-1)([avg_pool, max_pool])
    spatial_features = Conv2D(filters=3, kernel_size=(1, 1), activation='sigmoid')(spatial_features)
    
    # Multiply spatial and channel features
    combined_features = Multiply()([channel_features, spatial_features])
    
    # Additional branch with 1x1 convolution to ensure output channels align
    additional_branch = Conv2D(filters=3, kernel_size=(1, 1))(combined_features)
    
    # Add result to the main path and apply activation
    main_path = Add()([x, additional_branch])
    main_output = Activation('relu')(main_path)
    
    # Final classification
    final_output = GlobalAveragePooling2D()(main_output)
    final_output = Dense(10, activation='softmax')(final_output)
    
    # Construct model
    model = Model(inputs=inputs, outputs=final_output)
    
    return model