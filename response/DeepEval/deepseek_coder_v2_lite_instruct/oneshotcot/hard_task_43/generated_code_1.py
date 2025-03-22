import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape
from keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 2: Add convolutional layer (Block 1)
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 3: Add maxpooling layer (Block 1)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    
    # Define Block 1
    def block1(input_tensor):
        # Step 4.1: Add 1x1 average pooling layer
        path1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(input_tensor)
        # Step 4.2: Add 2x2 average pooling layer
        path2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        # Step 4.3: Add 4x4 average pooling layer
        path3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='same')(input_tensor)
        
        # Step 4.4: Concatenate the outputs of the pooling paths
        output_tensor = Concatenate()([path1, path2, path3])
        
        return output_tensor
    
    # Apply Block 1
    block1_output = block1(max_pooling1)
    
    # Step 5: Add batch normalization (Block 1)
    batch_norm1 = BatchNormalization()(block1_output)
    
    # Step 6: Add flatten layer (Block 1)
    flatten_layer = Flatten()(batch_norm1)
    
    # Step 7: Add fully connected layer (between Block 1 and Block 2)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    
    # Step 8: Reshape the output from Block 1 to 4D tensor for Block 2
    reshape_layer = Reshape((1, 1, 128))(dense_layer)
    
    # Step 9: Add convolutional layer (Block 2)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    
    # Define Block 2
    def block2(input_tensor):
        # Branch 1: 1x1 convolution
        branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Branch 2: 3x3 convolution
        branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Branch 3: 1x7 convolution followed by 7x1 convolution
        branch3_1 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3_2 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(branch3_1)
        # Branch 4: 3x3 convolution
        branch4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Branch 5: Average pooling
        branch5 = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(input_tensor)
        
        # Step 4.5: Concatenate the outputs of the branches
        output_tensor = Concatenate()([branch1, branch2, branch3_2, branch4, branch5])
        
        return output_tensor
    
    # Apply Block 2
    block2_output = block2(conv2)
    
    # Step 10: Add batch normalization (Block 2)
    batch_norm2 = BatchNormalization()(block2_output)
    
    # Step 11: Add flatten layer (Block 2)
    flatten_layer_block2 = Flatten()(batch_norm2)
    
    # Step 12: Add fully connected layers for classification
    dense1 = Dense(units=64, activation='relu')(flatten_layer_block2)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model