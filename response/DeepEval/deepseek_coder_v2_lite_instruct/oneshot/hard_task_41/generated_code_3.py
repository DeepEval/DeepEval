import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Reshape

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: Three parallel paths with different pooling layers
    def block1(input_tensor):
        # Path 1: 1x1 average pooling
        path1 = AveragePooling2D(pool_size=(1, 1), strides=1)(input_tensor)
        path1 = Flatten()(path1)
        
        # Path 2: 2x2 average pooling
        path2 = AveragePooling2D(pool_size=(2, 2), strides=2)(input_tensor)
        path2 = Flatten()(path2)
        
        # Path 3: 4x4 average pooling
        path3 = AveragePooling2D(pool_size=(4, 4), strides=4)(input_tensor)
        path3 = Flatten()(path3)
        
        # Dropout regularization
        paths = [path1, path2, path3]
        combined_path = Concatenate()(paths)
        dropout_path = Dropout(0.5)(combined_path)
        
        return dropout_path
    
    # Apply Block 1
    block1_output = block1(input_tensor=input_layer)
    
    # Fully connected layer and reshape for Block 2
    dense_layer = Dense(units=128, activation='relu')(block1_output)
    reshape_layer = Reshape((1, 1, 128))(dense_layer)  # Reshape to 4D tensor
    
    # Block 2: Multiple branches for feature extraction
    def block2(input_tensor):
        # Branch 1: 1x1 convolution
        branch1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)
        
        # Branch 2: <1x1 convolution
        branch2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)
        branch2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(branch2)
        
        # Branch 3: 3x3 convolution
        branch3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_tensor)
        
        # Branch 4: <1x1 convolution, 3x3 convolution, 3x3 convolution>
        branch4_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)
        branch4_2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch4_1)
        branch4_3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch4_2)
        
        # Branch 5: <average pooling, 1x1 convolution>
        branch5_1 = AveragePooling2D(pool_size=(2, 2), strides=2)(input_tensor)
        branch5_2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(branch5_1)
        
        # Concatenate all branches
        branches = [branch1, branch2, branch3, branch4_3, branch5_2]
        combined_branches = Concatenate()(branches)
        
        return combined_branches
    
    # Apply Block 2
    block2_output = block2(input_tensor=reshape_layer)
    
    # Flatten and add final layers
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model