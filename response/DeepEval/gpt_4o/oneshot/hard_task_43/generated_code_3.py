import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense, Reshape, MaxPooling2D

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: Three parallel paths with average pooling of different scales
    def block1(input_tensor):
        path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_tensor)
        path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_tensor)
        path3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_tensor)
        
        path1_flatten = Flatten()(path1)
        path2_flatten = Flatten()(path2)
        path3_flatten = Flatten()(path3)
        
        output_tensor = Concatenate()([path1_flatten, path2_flatten, path3_flatten])
        return output_tensor
    
    block1_output = block1(input_layer)
    
    # Fully connected layer and reshape between Block 1 and Block 2
    dense_intermediate = Dense(units=256, activation='relu')(block1_output)
    reshape_layer = Reshape((4, 4, 16))(dense_intermediate)  # Reshape to 4D tensor
    
    # Block 2: Three branches for feature extraction
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
        branch3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch3)
        
        # Concatenate all branches
        output_tensor = Concatenate()([branch1, branch2, branch3])
        return output_tensor
    
    block2_output = block2(reshape_layer)
    
    # Final classification through fully connected layers
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example of how to create the model
model = dl_model()
model.summary()