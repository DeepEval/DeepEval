import keras
import tensorflow as tf
from keras.layers import Input, Lambda, DepthwiseConv2D, BatchNormalization, Concatenate, Conv2D, AveragePooling2D, Dense, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Depthwise separable convolutions
    def block_1(input_tensor):
        # Split into three groups
        split_inputs = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Apply depthwise separable convolutions
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(split_inputs[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(split_inputs[1])
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(split_inputs[2])
        
        # Apply batch normalization
        norm1 = BatchNormalization()(conv1)
        norm2 = BatchNormalization()(conv2)
        norm3 = BatchNormalization()(conv3)
        
        # Concatenate the outputs
        output_tensor = Concatenate()([norm1, norm2, norm3])
        return output_tensor

    # Block 2: Multiple branches for feature extraction
    def block_2(input_tensor):
        # Branch 1: 1x1 conv and 3x3 conv
        branch1 = Concatenate()([
            Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor),
            Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        ])
        
        # Branch 2: 1x1 conv, 1x7 conv, 7x1 conv, and 3x3 conv
        branch2 = Concatenate()([
            Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor),
            Conv2D(filters=32, kernel_size=(1, 7), padding='same', activation='relu')(input_tensor),
            Conv2D(filters=32, kernel_size=(7, 1), padding='same', activation='relu')(input_tensor),
            Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        ])
        
        # Branch 3: Average pooling
        branch3 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(input_tensor)

        # Concatenate all branches
        output_tensor = Concatenate()([branch1, branch2, branch3])
        return output_tensor

    # Build the model
    block1_output = block_1(input_layer)
    block2_output = block_2(block1_output)

    # Fully connected layers for classification
    flatten = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model