import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, Reshape
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First block processing
    def first_block(input_tensor):
        # Four parallel branches
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        
        # Concatenate outputs
        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor
    
    # Apply first block
    first_block_output = first_block(input_layer)
    
    # Batch normalization and flattening
    batch_norm = BatchNormalization()(first_block_output)
    flatten_layer = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Second block processing
    def second_block(input_tensor):
        # Global average pooling
        gap = GlobalAveragePooling2D()(input_tensor)
        
        # Fully connected layers
        fc1 = Dense(units=128, activation='relu')(gap)
        fc2 = Dense(units=64, activation='relu')(fc1)
        
        # Reshape to match input shape
        reshape = Reshape((32, 32, 3))(fc2)
        
        # Element-wise multiplication with input
        output_tensor = keras.layers.Multiply()([input_tensor, reshape])
        return output_tensor
    
    # Apply second block
    second_block_output = second_block(first_block_output)
    
    # Final fully connected layer
    final_dense = Dense(units=10, activation='softmax')(second_block_output)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=final_dense)
    
    return model