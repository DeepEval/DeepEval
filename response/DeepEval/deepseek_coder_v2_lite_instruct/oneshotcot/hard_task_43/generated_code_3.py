import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 2: Add first specialized block (Block 1)
    def block1(input_tensor):
        # Path 1: 1x1 Average Pooling
        path1 = AveragePooling2D(pool_size=(1, 1), strides=1)(input_tensor)
        path1 = Flatten()(path1)
        
        # Path 2: 2x2 Average Pooling
        path2 = AveragePooling2D(pool_size=(2, 2), strides=2)(input_tensor)
        path2 = Flatten()(path2)
        
        # Path 3: 4x4 Average Pooling
        path3 = AveragePooling2D(pool_size=(4, 4), strides=4)(input_tensor)
        path3 = Flatten()(path3)
        
        # Concatenate outputs of the three paths
        output_tensor = Concatenate()([path1, path2, path3])
        return output_tensor
    
    block1_output = block1(input_layer)
    
    # Apply a fully connected layer between Block 1 and Block 2
    fc_layer = Dense(units=256, activation='relu')(block1_output)
    
    # Reshape the output of Block 1 to a 4-dimensional tensor for Block 2
    reshape_layer = Reshape((4, 4, 16))(fc_layer)  # Assuming 16 filters after reshape
    
    # Step 3: Add second specialized block (Block 2)
    def block2(input_tensor):
        # Branch 1: 1x1 Convolution
        branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Branch 2: 3x3 Convolution
        branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Branch 3: 1x7 Convolution followed by 7x1 Convolution
        branch3 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(branch3)
        # Branch 4: 3x3 Convolution
        branch4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Branch 5: Average Pooling
        branch5 = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(input_tensor)
        
        # Concatenate outputs of all branches
        output_tensor = Concatenate()([branch1, branch2, branch3, branch4, branch5])
        return output_tensor
    
    block2_output = block2(reshape_layer)
    
    # Step 4: Add batch normalization
    batch_norm = BatchNormalization()(block2_output)
    
    # Step 5: Add flatten layer
    flatten_layer = Flatten()(batch_norm)
    
    # Step 6: Add dense layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Step 7: Add second dense layer
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Step 8: Add output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Step 9: Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model