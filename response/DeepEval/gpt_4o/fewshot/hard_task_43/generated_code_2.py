import keras
import tensorflow as tf
from keras.layers import Input, AveragePooling2D, Conv2D, Flatten, Concatenate, Dense, Reshape

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block_1(input_tensor):
        # Three parallel paths with different average pooling scales
        pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(pool1)
        
        pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(pool2)
        
        pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(pool3)
        
        # Concatenate all flattened outputs
        output_tensor = Concatenate()([flatten1, flatten2, flatten3])
        return output_tensor

    def block_2(input_tensor):
        # Three branches for feature extraction
        # Branch 1: 1x1 convolution followed by 3x3 convolution
        branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
        
        # Branch 2: 1x1 convolution, 1x7 convolution, 7x1 convolution, and 3x3 convolution
        branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
        
        # Branch 3: Average pooling
        branch3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        
        # Concatenate all branch outputs
        output_tensor = Concatenate()([branch1, branch2, branch3])
        return output_tensor

    # Block 1 processing
    block1_output = block_1(input_tensor=input_layer)
    
    # Fully connected layer and reshape
    fc = Dense(units=128, activation='relu')(block1_output)
    reshaped = Reshape(target_shape=(8, 8, 2))(fc)  # Adjusted the shape for the next block

    # Block 2 processing
    block2_output = block_2(input_tensor=reshaped)

    # Final classification layers
    flatten = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model