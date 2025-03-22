import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Multiply, Activation, Concatenate, AveragePooling2D, Lambda
from keras import backend as K

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    def block1(x):
        # Path 1: Global Average Pooling followed by two fully connected layers
        gap = GlobalAveragePooling2D()(x)
        dense1 = Dense(32, activation='relu')(gap)
        dense2 = Dense(32, activation='relu')(dense1)
        path1 = dense2
        
        # Path 2: Global Max Pooling followed by two fully connected layers
        gmp = GlobalMaxPooling2D()(x)
        dense1 = Dense(32, activation='relu')(gmp)
        dense2 = Dense(32, activation='relu')(dense1)
        path2 = dense2
        
        # Merge paths
        merge = Add()([path1, path2])
        
        # Channel attention weights
        channel_weights = Dense(K.int_shape(x)[-1], activation='sigmoid')(merge)
        
        # Apply weights to original features
        weighted_features = Multiply()([x, channel_weights])
        
        return weighted_features
    
    block1_output = block1(input_layer)
    
    # Block 2
    def block2(x):
        # Extract spatial features
        avg_pool = AveragePooling2D(pool_size=(7, 7))(x)
        max_pool = MaxPooling2D(pool_size=(7, 7))(x)
        
        # Concatenate along the channel dimension
        concat = Concatenate(axis=-1)([avg_pool, max_pool])
        
        # 1x1 convolution and sigmoid activation
        conv = Conv2D(3, kernel_size=(1, 1), activation='sigmoid')(concat)
        
        # Normalize features
        normalized_features = Multiply()([x, conv])
        
        return normalized_features
    
    block2_output = block2(block1_output)
    
    # Additional branch to ensure output channels align with input channels
    branch = Conv2D(3, kernel_size=(1, 1), activation='linear')(block1_output)
    final_output = Add()([block2_output, branch])
    
    # Flatten and fully connected layers
    flatten = Flatten()(final_output)
    dense1 = Dense(128, activation='relu')(flatten)
    dense2 = Dense(64, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model