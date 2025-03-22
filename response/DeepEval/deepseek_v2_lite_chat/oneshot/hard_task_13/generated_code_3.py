import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D

def dl_model():     
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Define the first block
    def block1(input_tensor):
        # Four parallel branches
        branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch4 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='valid')(input_tensor)
        
        # Concatenate the outputs of these paths
        concatenated = Concatenate()( [branch1, branch2, branch3, branch4] )
        
        return concatenated
    
    # Define the second block
    def block2(input_tensor):
        # Global average pooling
        pooled_features = GlobalAveragePooling2D()(input_tensor)
        
        # Fully connected layers
        dense1 = Dense(units=128, activation='relu')(pooled_features)
        dense2 = Dense(units=64, activation='relu')(dense1)
        
        # Output layer
        output_layer = Dense(units=10, activation='softmax')(dense2)
        
        return output_layer
    
    # Apply the blocks
    block_output = block1(input_layer)
    model_output = block2(block_output)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=model_output)
    
    return model