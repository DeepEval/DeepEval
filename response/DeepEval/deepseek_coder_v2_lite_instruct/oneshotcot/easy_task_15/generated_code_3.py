import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Dropout, Flatten, Dense
from keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 2: Add convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    
    # Step 3: Add average pooling layer
    avg_pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)
    
    # Step 4: Add dropout layer
    dropout1 = Dropout(0.5)(avg_pool1)
    
    # Step 5: Define the specialized block
    def specialized_block(input_tensor):
        # Step 5.1: Add 3x3 convolutional layer
        conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(input_tensor)
        
        # Step 5.2: Add first 1x1 convolutional layer
        conv1x1_1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_tensor)
        
        # Step 5.3: Add second 1x1 convolutional layer
        conv1x1_2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_tensor)
        
        # Step 5.4: Add average pooling layer
        avg_pool = AveragePooling2D(pool_size=(2, 2))(input_tensor)
        
        # Step 5.5: Concatenate the outputs of the paths
        output_tensor = keras.layers.concatenate([conv3x3, conv1x1_1, conv1x1_2, avg_pool], axis=-1)
        
        return output_tensor
    
    # Apply the block twice
    block_output = specialized_block(dropout1)
    block_output = specialized_block(block_output)
    
    # Step 6: Add global average pooling layer
    global_avg_pool = keras.layers.GlobalAveragePooling2D()(block_output)
    
    # Step 7: Add flatten layer
    flatten_layer = Flatten()(global_avg_pool)
    
    # Step 8: Add dense layer for final classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Step 9: Build the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model