import keras
from keras.layers import Input, Lambda, SeparableConv2D, BatchNormalization, Concatenate, AveragePooling2D, Flatten, Dense, Conv2D
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First Block: Split input and apply depthwise separable convolutions
    def split_and_conv(input_tensor):
        # Split along the last dimension
        splits = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)

        # Applying depthwise separable convolutions with different kernel sizes
        conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(splits[0])
        conv3 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(splits[1])
        conv5 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(splits[2])

        # Batch normalization
        norm1 = BatchNormalization()(conv1)
        norm3 = BatchNormalization()(conv3)
        norm5 = BatchNormalization()(conv5)

        # Concatenate the outputs
        return Concatenate()([norm1, norm3, norm5])
    
    block1_output = split_and_conv(input_layer)
    
    # Second Block: Multiple branches for feature extraction
    def multi_branch(input_tensor):
        # Branch 1: 1x1 convolution followed by 3x3 convolution
        branch1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        branch1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(branch1)

        # Branch 2: 1x1 convolution, 1x7 convolution, 7x1 convolution, 3x3 convolution
        branch2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(1, 7), activation='relu', padding='same')(branch2)
        branch2 = Conv2D(filters=64, kernel_size=(7, 1), activation='relu', padding='same')(branch2)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(branch2)

        # Branch 3: Average pooling
        branch3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)

        # Concatenate the outputs
        return Concatenate()([branch1, branch2, branch3])
    
    block2_output = multi_branch(block1_output)

    # Fully connected layers for final classification
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model