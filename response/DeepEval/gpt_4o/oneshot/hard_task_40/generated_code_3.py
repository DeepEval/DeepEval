import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Dropout

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First Block: Average Pooling and Concatenation
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)
    
    flat1 = Flatten()(avg_pool1)
    flat2 = Flatten()(avg_pool2)
    flat3 = Flatten()(avg_pool3)
    
    concat_pool = Concatenate()([flat1, flat2, flat3])
    fc_layer = Dense(units=256, activation='relu')(concat_pool)
    
    # Reshape to prepare for the second block
    reshaped_output = Reshape((7, 7, 4))(fc_layer)  # Assuming reshaped dimensions fit

    # Second Block: Multi-Scale Feature Extraction
    def multi_scale_block(input_tensor):
        # Path 1: 1x1 Convolution
        path1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_tensor)
        path1 = Dropout(0.3)(path1)
        
        # Path 2: 1x1 Convolution -> 3x3 Convolution -> 3x3 Convolution
        path2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(path2)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(path2)
        path2 = Dropout(0.3)(path2)
        
        # Path 3: 1x1 Convolution -> 3x3 Convolution
        path3 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(path3)
        path3 = Dropout(0.3)(path3)
        
        # Path 4: Average Pooling -> 1x1 Convolution
        path4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        path4 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(path4)
        path4 = Dropout(0.3)(path4)

        # Concatenate all paths
        output_tensor = Concatenate()([path1, path2, path3, path4])
        
        return output_tensor
    
    block_output = multi_scale_block(reshaped_output)
    
    # Fully Connected Layers
    flatten_layer = Flatten()(block_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model