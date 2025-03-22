import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Permute, Reshape
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Reshape input to (height, width, groups, channels_per_group)
    reshaped = Reshape(target_shape=(32, 32, 3, 1))(input_layer)
    
    # Permute dimensions to shuffle channels
    permuted = Permute((4, 1, 2, 3))(reshaped)
    
    # Reshape back to original shape
    original_shape = (32, 32, 3)
    
    # Define a block for feature extraction
    def block(input_tensor):
        # Paths for different convolution sizes
        path1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # MaxPooling layer
        path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        
        # Concatenate outputs of different paths
        output_tensor = Concatenate()([path1, path2, path3, path4])
        
        return output_tensor
    
    # Apply the block to the permuted tensor
    block_output = block(permuted)
    
    # Batch Normalization
    batch_norm = BatchNormalization()(block_output)
    
    # Flatten layer
    flatten_layer = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model