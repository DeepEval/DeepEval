import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Reshape

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    def block1(input_tensor):
        # Three parallel paths with different average pooling layers
        path1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(input_tensor)
        path2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        path3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='same')(input_tensor)
        
        # Flatten each path and apply dropout
        path1_flat = Flatten()(path1)
        path2_flat = Flatten()(path2)
        path3_flat = Flatten()(path3)
        
        path1_dropout = Dropout(0.2)(path1_flat)
        path2_dropout = Dropout(0.2)(path2_flat)
        path3_dropout = Dropout(0.2)(path3_flat)
        
        # Concatenate the outputs of the three paths
        concatenated = Concatenate()([path1_dropout, path2_dropout, path3_dropout])
        return concatenated
    
    block1_output = block1(input_layer)
    
    # Fully connected layer and reshape to prepare for Block 2
    reshape_layer = Reshape((-1,))(block1_output)
    fc_layer = Dense(units=128, activation='relu')(reshape_layer)
    
    # Block 2
    def block2(input_tensor):
        # Four branches in Block 2
        branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
        
        branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
        
        branch4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        branch4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch4)
        
        # Concatenate the outputs of the four branches
        concatenated = Concatenate()([branch1, branch2, branch3, branch4])
        return concatenated
    
    block2_output = block2(fc_layer)
    
    # Batch normalization and flatten
    batch_norm = BatchNormalization()(block2_output)
    flatten_layer = Flatten()(batch_norm)
    
    # Two fully connected layers for final classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model