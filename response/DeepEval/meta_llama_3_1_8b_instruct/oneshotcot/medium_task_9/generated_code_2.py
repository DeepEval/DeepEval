import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D

def dl_model():
    
    # Define the basic block
    def basic_block(input_tensor):
        
        # Main path
        conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        bath_norm = BatchNormalization()(conv)
        
        # Branch path
        conv_branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        
        # Feature fusion
        output_tensor = keras.layers.Add()([conv, conv_branch])
        
        return output_tensor
    
    # Define the initial convolutional layer
    input_layer = Input(shape=(32, 32, 3))
    initial_conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Apply the basic block twice
    block1 = basic_block(initial_conv)
    block2 = basic_block(block1)
    
    # Define the feature extraction path
    conv_extract = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(block2)
    
    # Combine the outputs
    output_tensor = keras.layers.Add()([block2, conv_extract])
    
    # Apply average pooling and flatten
    avg_pool = AveragePooling2D(pool_size=(8, 8))(output_tensor)
    flatten_layer = Flatten()(avg_pool)
    
    # Define the final fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model