import keras
from keras.layers import Input, Lambda, DepthwiseConv2D, BatchNormalization, Concatenate, Flatten, Dense, AveragePooling2D
import tensorflow as tf

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # First block: split input into three groups and apply depthwise separable convolutions
    split_tensors = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Applying depthwise separable convolutions with different kernel sizes
    def depthwise_block(input_tensor, kernel_size):
        x = DepthwiseConv2D(kernel_size=(kernel_size, kernel_size), padding='same', activation='relu')(input_tensor)
        x = BatchNormalization()(x)
        return x
    
    conv1 = depthwise_block(split_tensors[0], kernel_size=1)
    conv2 = depthwise_block(split_tensors[1], kernel_size=3)
    conv3 = depthwise_block(split_tensors[2], kernel_size=5)
    
    # Concatenate outputs of the first block
    first_block_output = Concatenate()([conv1, conv2, conv3])

    # Second block: multiple branches for feature extraction
    branch1 = Concatenate()([
        DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(first_block_output),
        DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(first_block_output)
    ])
    
    branch2 = Concatenate()([
        DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(first_block_output),
        DepthwiseConv2D(kernel_size=(1, 7), padding='same', activation='relu')(first_block_output),
        DepthwiseConv2D(kernel_size=(7, 1), padding='same', activation='relu')(first_block_output),
        DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(first_block_output)
    ])
    
    branch3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(first_block_output)
    
    # Concatenate outputs of the second block
    second_block_output = Concatenate()([branch1, branch2, branch3])

    # Flatten and fully connected layers for classification
    flatten_layer = Flatten()(second_block_output)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # 10 classes for CIFAR-10

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model