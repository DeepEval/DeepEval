import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D, Reshape

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    def block1(input_tensor):
        # 1x1 average pooling
        path1 = AveragePooling2D(pool_size=(1, 1), strides=1)(input_tensor)
        path1 = Flatten()(path1)
        
        # 2x2 average pooling
        path2 = AveragePooling2D(pool_size=(2, 2), strides=2)(input_tensor)
        path2 = Flatten()(path2)
        
        # 4x4 average pooling
        path3 = AveragePooling2D(pool_size=(4, 4), strides=4)(input_tensor)
        path3 = Flatten()(path3)
        
        # Concatenate the outputs of the three paths
        output_tensor = Concatenate()([path1, path2, path3])
        
        return output_tensor
    
    block1_output = block1(input_layer)
    
    # Fully connected layer between Block 1 and Block 2
    fc_layer = Dense(units=128, activation='relu')(block1_output)
    
    # Reshape the output from Block 1 to 4-dimensional tensor
    reshape_layer = Reshape((1, 1, 128))(fc_layer)
    
    # Block 2
    def block2(input_tensor):
        # First branch: 1x1 convolution
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Second branch: 3x3 convolution
        path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Third branch: 1x7 convolution followed by 7x1 convolution
        path3 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3)
        
        # Fourth branch: average pooling
        path4 = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(input_tensor)
        
        # Concatenate the outputs of all branches
        output_tensor = Concatenate()([path1, path2, path3, path4])
        
        return output_tensor
    
    block2_output = block2(reshape_layer)
    
    # Flatten the output from Block 2
    flatten_layer = Flatten()(block2_output)
    
    # Fully connected layers for classification
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model