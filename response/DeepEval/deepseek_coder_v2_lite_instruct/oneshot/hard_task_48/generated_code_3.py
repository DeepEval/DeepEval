import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, AveragePooling2D
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    def block1(x):
        # Split the input into three groups
        splits = tf.split(x, num_or_size_splits=3, axis=3)
        
        # Extract features through separable convolutional layers with different kernel sizes
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(splits[0])
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(splits[1])
        conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(splits[2])
        
        # Concatenate the outputs of the three groups
        concatenated = Concatenate(axis=3)([conv1x1, conv3x3, conv5x5])
        
        # Batch normalization
        normalized = BatchNormalization()(concatenated)
        
        return normalized
    
    block1_output = block1(input_layer)
    
    # Block 2
    def block2(x):
        # Path 1: 1x1 convolution
        path1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x)
        
        # Path 2: 3x3 average pooling followed by 1x1 convolution
        path2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(AveragePooling2D(pool_size=(3, 3), strides=1)(x))
        
        # Path 3: 1x1 convolution followed by 1x3 and 3x1 convolutions
        path3_conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x)
        path3_1x3 = Conv2D(filters=64, kernel_size=(1, 3), activation='relu')(path3_conv1x1)
        path3_3x1 = Conv2D(filters=64, kernel_size=(3, 1), activation='relu')(path3_conv1x1)
        path3_concat = Concatenate(axis=3)([path3_1x3, path3_3x1])
        
        # Path 4: 1x1 convolution followed by 3x3 convolution, then 1x3 and 3x1 convolutions
        path4_conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x)
        path4_3x3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(path4_conv1x1)
        path4_1x3 = Conv2D(filters=64, kernel_size=(1, 3), activation='relu')(path4_conv1x1)
        path4_3x1 = Conv2D(filters=64, kernel_size=(3, 1), activation='relu')(path4_conv1x1)
        path4_concat = Concatenate(axis=3)([path4_3x3, path4_1x3, path4_3x1])
        
        # Concatenate the outputs of the four paths
        concatenated = Concatenate(axis=3)([path1, path2, path3_concat, path4_concat])
        
        return concatenated
    
    block2_output = block2(block1_output)
    
    # Flatten the result
    flattened = Flatten()(block2_output)
    
    # Fully connected layer
    dense1 = Dense(units=256, activation='relu')(flattened)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model