import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense, Reshape

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: Three parallel paths with average pooling
    def block1(input_tensor):
        path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_tensor)
        path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_tensor)
        path3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_tensor)
        
        # Flatten each path to a 1D vector
        path1_flat = Flatten()(path1)
        path2_flat = Flatten()(path2)
        path3_flat = Flatten()(path3)
        
        # Concatenate the flattened outputs
        output_tensor = Concatenate()([path1_flat, path2_flat, path3_flat])
        return output_tensor
    
    block1_output = block1(input_layer)
    
    # Fully connected layer between Block 1 and Block 2
    fc_layer = Dense(units=256, activation='relu')(block1_output)
    
    # Reshape to a 4D tensor for Block 2 processing
    reshape_layer = Reshape((8, 8, 4))(fc_layer)  # Assuming a reshape to a size compatible with subsequent convolutions
    
    # Block 2: Feature extraction with three branches
    def block2(input_tensor):
        # Branch 1: <1x1 convolution, 3x3 convolution>
        branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
        
        # Branch 2: <1x1 convolution, 1x7 convolution, 7x1 convolution, 3x3 convolution>
        branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
        
        # Branch 3: Average pooling
        branch3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        
        # Concatenate outputs from all branches
        output_tensor = Concatenate()([branch1, branch2, branch3])
        return output_tensor
    
    block2_output = block2(reshape_layer)
    
    # Flatten the output for fully connected layers
    flatten_layer = Flatten()(block2_output)
    
    # Fully connected layers for classification
    fc2 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(fc2)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model