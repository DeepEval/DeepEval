import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense, Reshape, BatchNormalization

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: Three parallel paths with different average pooling layers
    def block1(input_tensor):
        path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_tensor)
        path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_tensor)
        path3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3])
        return output_tensor
    
    # Apply Block 1 to the input
    block1_output = block1(input_tensor=input_layer)
    
    # Flatten the output of Block 1
    flatten1 = Flatten()(block1_output)
    
    # Add a fully connected layer after Block 1
    fc1 = Dense(units=128, activation='relu')(flatten1)
    
    # Reshape the output to prepare for Block 2
    reshape_output = Reshape((1, 1, 128))(fc1)
    
    # Block 2: Three branches with different configurations
    def block2(input_tensor):
        # First branch: 1x1 convolution followed by 3x3 convolution
        branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
        
        # Second branch: 1x1 convolution, 1x7 convolution, 7x1 convolution, and 3x3 convolution
        branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
        
        # Third branch: Average pooling
        branch3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        
        # Concatenate the outputs of all branches
        output_tensor = Concatenate()([branch1, branch2, branch3])
        return output_tensor
    
    # Apply Block 2 to the reshaped output of Block 1
    block2_output = block2(input_tensor=reshape_output)
    
    # Flatten the output of Block 2
    flatten2 = Flatten()(block2_output)
    
    # Add batch normalization before the final fully connected layers
    batch_norm = BatchNormalization()(flatten2)
    
    # Two fully connected layers for classification
    fc2 = Dense(units=128, activation='relu')(batch_norm)
    output_layer = Dense(units=10, activation='softmax')(fc2)
    
    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model