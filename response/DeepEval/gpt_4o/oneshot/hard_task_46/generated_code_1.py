import keras
from keras.layers import Input, Lambda, SeparableConv2D, Concatenate, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First Block: Split channels and apply SeparableConv2D
    def first_block(input_tensor):
        # Split the input into 3 parts along the channel dimension
        split_channels = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Apply separable convolutions with different kernel sizes
        conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_channels[0])
        conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_channels[1])
        conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_channels[2])
        
        # Concatenate the outputs
        output_tensor = Concatenate()([conv1, conv2, conv3])
        
        return output_tensor

    # Second Block: Enhanced feature extraction with multiple branches
    def second_block(input_tensor):
        # Branch 1: Simple 3x3 convolution
        branch1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        
        # Branch 2: 1x1 convolution followed by two 3x3 convolutions
        branch2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
        
        # Branch 3: Max pooling
        branch3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)

        # Concatenate all branches
        output_tensor = Concatenate()([branch1, branch2, branch3])
        
        return output_tensor

    # Build the model architecture
    x = first_block(input_layer)
    x = second_block(x)
    
    # Global Average Pooling
    gap = GlobalAveragePooling2D()(x)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(gap)
    
    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model