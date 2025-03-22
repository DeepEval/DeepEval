import keras
from keras.layers import Input, Lambda, Conv2D, BatchNormalization, Concatenate, AveragePooling2D, Dense, GlobalAveragePooling2D
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        # Split the input into three groups
        groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Depthwise separable convolutions
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(groups[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(groups[1])
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(groups[2])
        
        # Batch normalization
        conv1 = BatchNormalization()(conv1)
        conv2 = BatchNormalization()(conv2)
        conv3 = BatchNormalization()(conv3)
        
        # Concatenate the outputs
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    def block_2(input_tensor):
        # First branch: 1x1 convolution followed by 3x3 convolution
        branch1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_tensor)
        branch1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch1)
        
        # Second branch: 1x1 convolution, 1x7 convolution, 7x1 convolution, and 3x3 convolution
        branch2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(1, 7), activation='relu')(branch2)
        branch2 = Conv2D(filters=64, kernel_size=(7, 1), activation='relu')(branch2)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch2)
        
        # Third branch: average pooling
        branch3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        branch3 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(branch3)
        
        # Concatenate the outputs of all branches
        output_tensor = Concatenate()([branch1, branch2, branch3])
        return output_tensor

    # Apply the first block to the input
    block1_output = block_1(input_tensor=input_layer)
    
    # Apply the second block to the output of the first block
    block2_output = block_2(input_tensor=block1_output)
    
    # Flatten the output of the second block
    flatten_layer = Flatten()(block2_output)
    
    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model