import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D, Reshape

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First Block
    def block1(input_tensor):
        # Three parallel paths with different average pooling layers
        path1 = AveragePooling2D(pool_size=(1, 1), strides=1)(input_tensor)
        path2 = AveragePooling2D(pool_size=(2, 2), strides=2)(input_tensor)
        path3 = AveragePooling2D(pool_size=(4, 4), strides=4)(input_tensor)
        
        # Concatenate the outputs of the three paths
        output_tensor = Concatenate()([path1, path2, path3])
        
        # Flatten the concatenated output
        output_tensor = Flatten()(output_tensor)
        
        return output_tensor
    
    # Apply fully connected layer between Block 1 and Block 2
    fc_layer = Dense(units=128, activation='relu')(block1(input_tensor=input_layer))
    
    # Reshape the output from Block 1 to a 4-dimensional tensor
    reshape_layer = Reshape((1, 1, 128))(fc_layer)
    
    # Second Block
    def block2(input_tensor):
        # Three branches for feature extraction
        
        # First branch: 1x1 convolution followed by 3x3 convolution
        branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
        
        # Second branch: 1x1 convolution, 1x7 convolution, 7x1 convolution, and 3x3 convolution
        branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
        
        # Third branch: average pooling
        branch3 = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(input_tensor)
        
        # Concatenate the outputs from all branches
        output_tensor = Concatenate()([branch1, branch2, branch3])
        
        return output_tensor
    
    # Apply batch normalization and flatten the result
    block2_output = block2(input_tensor=reshape_layer)
    batch_norm = BatchNormalization()(block2_output)
    flatten_layer = Flatten()(batch_norm)
    
    # Fully connected layers for classification
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model