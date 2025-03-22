import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Reshape, Dense

def dl_model():
    # Define the input layer with shape (28, 28, 1)
    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Three parallel paths with average pooling layers
    def block1(input_tensor):
        path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_tensor)
        path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_tensor)
        path3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_tensor)
        
        # Flatten the output of each path and concatenate
        flatten1 = Flatten()(path1)
        flatten2 = Flatten()(path2)
        flatten3 = Flatten()(path3)
        output_tensor = Concatenate()([flatten1, flatten2, flatten3])
        
        return output_tensor
    
    block1_output = block1(input_layer)
    
    # Apply a fully connected layer and reshape the output to 4D
    fc1 = Dense(units=128, activation='relu')(block1_output)
    reshape_layer = Reshape((4, 16))(fc1)
    
    # Block 2: Three branches for feature extraction
    def block2(input_tensor):
        branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(branch3)
        branch3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(branch3)
        branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
        branch4 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_tensor)
        
        # Concatenate the outputs from all branches
        output_tensor = Concatenate()([branch1, branch2, branch3, branch4])
        
        return output_tensor
    
    block2_output = block2(reshape_layer)
    
    # Batch normalization and flatten the output of block 2
    bath_norm = BatchNormalization()(block2_output)
    flatten_layer = Flatten()(bath_norm)
    
    # Two fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model