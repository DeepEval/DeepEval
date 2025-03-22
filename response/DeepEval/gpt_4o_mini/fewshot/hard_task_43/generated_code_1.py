import keras
import tensorflow as tf
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Conv2D

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    def block_1(input_tensor):
        avgpool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(avgpool1)
        
        avgpool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(avgpool2)
        
        avgpool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(avgpool3)
        
        output_tensor = Concatenate()([flatten1, flatten2, flatten3])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    
    # Fully connected layer after Block 1
    dense = Dense(units=64, activation='relu')(block1_output)
    
    # Reshape to prepare for Block 2
    reshaped = Reshape(target_shape=(4, 4, 4))(dense)

    # Block 2
    def block_2(input_tensor):
        # Branch 1: 1x1 conv followed by 3x3 conv
        branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)

        # Branch 2: 1x1 conv, 1x7 conv, 7x1 conv, and 3x3 conv
        branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)

        # Branch 3: Average pooling
        branch3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)

        # Concatenate outputs from all branches
        output_tensor = Concatenate()([branch1, branch2, branch3])
        return output_tensor

    block2_output = block_2(input_tensor=reshaped)

    # Final layers for classification
    flatten = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model