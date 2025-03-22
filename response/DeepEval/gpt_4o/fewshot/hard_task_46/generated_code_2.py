import keras
import tensorflow as tf
from keras.layers import Input, Lambda, Concatenate, SeparableConv2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        # Split input along the channel dimension into three groups
        groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Apply separable convolutions with different kernel sizes
        conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(groups[0])
        conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(groups[1])
        conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(groups[2])
        
        # Concatenate outputs
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    def block_2(input_tensor):
        # Branch 1: 3x3 convolution
        branch1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        
        # Branch 2: 1x1 convolution followed by two 3x3 convolutions
        branch2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
        
        # Branch 3: Max pooling
        branch3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        
        # Concatenate outputs from all branches
        output_tensor = Concatenate()([branch1, branch2, branch3])
        return output_tensor

    # Construct the full model
    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)

    # Global average pooling
    gap = GlobalAveragePooling2D()(block2_output)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(gap)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model