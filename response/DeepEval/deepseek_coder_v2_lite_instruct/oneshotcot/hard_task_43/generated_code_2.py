import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense, Reshape, BatchNormalization

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    def block1(input_tensor):
        # 1x1 Average Pooling
        path1 = AveragePooling2D(pool_size=(1, 1), strides=1)(input_tensor)
        path1 = Flatten()(path1)
        
        # 2x2 Average Pooling
        path2 = AveragePooling2D(pool_size=(2, 2), strides=2)(input_tensor)
        path2 = Flatten()(path2)
        
        # 4x4 Average Pooling
        path3 = AveragePooling2D(pool_size=(4, 4), strides=4)(input_tensor)
        path3 = Flatten()(path3)
        
        # Concatenate the outputs of the three paths
        output_tensor = Concatenate()([path1, path2, path3])
        
        return output_tensor
    
    block1_output = block1(input_tensor=input_layer)
    
    # Fully connected layer after Block 1
    dense_after_block1 = Dense(units=256, activation='relu')(block1_output)
    
    # Reshape Block 1 output to 4D tensor
    reshape_layer = Reshape((4, 4, 16))(dense_after_block1)
    
    # Block 2
    def block2(input_tensor):
        # 1x1 Convolution
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # 3x3 Convolution
        path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # 1x7 and 7x1 Convolution
        path3 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3)
        
        # 3x3 Convolution
        path4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Average Pooling
        path5 = AveragePooling2D(pool_size=(3, 3), strides=3)(input_tensor)
        path5 = Flatten()(path5)
        
        # Concatenate the outputs of the branches
        output_tensor = Concatenate()([path1, path2, path3, path4, path5])
        
        return output_tensor
    
    block2_output = block2(input_tensor=reshape_layer)
    
    # Batch Normalization
    batch_norm = BatchNormalization()(block2_output)
    
    # Flatten the output
    flatten_layer = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model