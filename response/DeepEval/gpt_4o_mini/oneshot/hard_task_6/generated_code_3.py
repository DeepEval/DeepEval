import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Lambda, Reshape, Permute, DepthwiseConv2D, Flatten, Dense

def block1(input_tensor):
    # Split the input into three groups
    split_tensors = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
    conv_tensors = [Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), activation='relu')(split) for split in split_tensors]
    # Concatenate the outputs from the three convolutions
    return Concatenate(axis=-1)(conv_tensors)

def block2(input_tensor):
    # Get the shape of the features and reshape
    shape = tf.shape(input_tensor)
    reshaped_tensor = Reshape((shape[1], shape[2], 3, shape[3] // 3))(input_tensor)
    # Permute dimensions to achieve channel shuffling
    permuted_tensor = Permute((0, 1, 3, 2))(reshaped_tensor)
    # Reshape back to original shape
    return Reshape((shape[1], shape[2], shape[3]))(permuted_tensor)

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Block 1
    block1_output = block1(input_layer)
    
    # Block 2
    block2_output = block2(block1_output)
    
    # Block 3: Depthwise separable convolution
    block3_output = DepthwiseConv2D(kernel_size=(3, 3), activation='relu', padding='same')(block2_output)
    
    # Repeat Block 1
    repeated_block1_output = block1(block3_output)
    
    # Branch path with average pooling
    branch_output = AveragePooling2D(pool_size=(2, 2))(input_layer)
    
    # Concatenate the main path and the branch path
    concatenated_output = Concatenate()([repeated_block1_output, branch_output])
    
    # Fully connected layer
    flatten_layer = Flatten()(concatenated_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # CIFAR-10 has 10 classes
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model