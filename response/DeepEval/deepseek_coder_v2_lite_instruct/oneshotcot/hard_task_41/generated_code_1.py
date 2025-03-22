import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Reshape

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 2: Add convolutional layer (Block 1)
    def block1(input_tensor):
        # Three parallel paths with different pooling sizes
        path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        path3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        
        # Flatten each path and apply dropout
        path1_flat = Flatten()(path1)
        path2_flat = Flatten()(path2)
        path3_flat = Flatten()(path3)
        
        path1_dropout = Dropout(0.2)(path1_flat)
        path2_dropout = Dropout(0.2)(path2_flat)
        path3_dropout = Dropout(0.2)(path3_flat)
        
        # Concatenate the flattened and dropout paths
        concatenated = Concatenate()([path1_dropout, path2_dropout, path3_dropout])
        return concatenated
    
    block1_output = block1(input_layer)
    
    # Step 3: Fully connected layer and reshape (Transform output of Block 1 to 4D tensor)
    reshape_layer = Reshape((14, 14, 1))(block1_output)
    fc_layer = Dense(256, activation='relu')(reshape_layer)
    
    # Step 4: Add Block 2
    def block2(input_tensor):
        # Four branches
        branch1 = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        branch4 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        branch5 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        
        # Concatenate the branches
        concatenated = Concatenate()([branch1, branch2, branch3, branch4, branch5])
        return concatenated
    
    block2_output = block2(fc_layer)
    
    # Step 5: Batch normalization
    batch_norm = BatchNormalization()(block2_output)
    
    # Step 6: Flatten
    flattened = Flatten()(batch_norm)
    
    # Step 7: Add dense layer
    dense1 = Dense(128, activation='relu')(flattened)
    
    # Step 8: Add dense layer
    dense2 = Dense(64, activation='relu')(dense1)
    
    # Step 9: Output layer
    output_layer = Dense(10, activation='softmax')(dense2)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model