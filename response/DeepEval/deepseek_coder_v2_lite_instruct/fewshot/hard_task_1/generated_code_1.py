import keras
from keras.layers import Input, Conv2D, Add, Activation, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Multiply, Concatenate, Reshape
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        # Path 1: Global Average Pooling followed by two fully connected layers
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=64, activation='relu')(avg_pool)
        dense2 = Dense(units=32, activation='relu')(dense1)
        
        # Path 2: Global Max Pooling followed by two fully connected layers
        max_pool = GlobalMaxPooling2D()(input_tensor)
        dense_max1 = Dense(units=64, activation='relu')(max_pool)
        dense_max2 = Dense(units=32, activation='relu')(dense_max1)
        
        # Add the outputs of both paths
        added = Add()([dense2, dense_max2])
        
        # Generate channel attention weights
        reshaped = Reshape((1, 1, 64))(added)
        activation_layer = Activation('sigmoid')(reshaped)
        
        # Apply channel attention weights to the original features
        attention_applied = Multiply()([input_tensor, activation_layer])
        
        return attention_applied

    def block_2(input_tensor):
        # Apply average pooling and max pooling separately
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        max_pool = GlobalMaxPooling2D()(input_tensor)
        
        # Concatenate the outputs along the channel dimension
        concatenated = Concatenate(axis=-1)([avg_pool, max_pool])
        
        # 1x1 convolution and sigmoid activation to normalize the features
        conv1x1 = Conv2D(filters=3, kernel_size=(1, 1), activation='sigmoid')(concatenated)
        
        # Multiply element-wise with the channel dimension features from Block 1
        normalized_features = Multiply()([input_tensor, conv1x1])
        
        return normalized_features

    # Apply Block 1
    block1_output = block_1(input_tensor=input_layer)
    
    # Apply Block 2
    block2_output = block_2(input_tensor=block1_output)
    
    # Additional branch with a 1x1 convolutional layer to ensure output channels align with input channels
    final_branch = Conv2D(filters=3, kernel_size=(1, 1), activation='relu')(block2_output)
    
    # Add the main path and the final branch
    added_final = Add()([block2_output, final_branch])
    
    # Flatten the final output
    flattened = Flatten()(added_final)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flattened)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model