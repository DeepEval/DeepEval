import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense, Lambda, Reshape, Permute
import tensorflow as tf

def block_1(input_tensor):
    # Split the input into three groups
    split_tensors = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
    
    # Apply 1x1 convolution to each group
    conv_outputs = [Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), activation='relu')(split) for split in split_tensors]
    
    # Concatenate the outputs of the three convolutions
    return Concatenate()([*conv_outputs])

def block_2(input_tensor):
    # Get the shape of the input tensor
    shape = tf.shape(input_tensor)
    height, width, channels = shape[1], shape[2], shape[3]
    
    # Reshape the tensor to (batch_size, height, width, groups, channels_per_group)
    groups = 3
    channels_per_group = channels // groups
    reshaped_tensor = Reshape((height, width, groups, channels_per_group))(input_tensor)

    # Permute to swap the last two dimensions
    permuted_tensor = Permute((1, 2, 4, 3))(reshaped_tensor)

    # Reshape back to the original shape
    return Reshape((height, width, channels))(permuted_tensor)

def block_3(input_tensor):
    # Apply 3x3 depthwise separable convolution
    return Conv2D(filters=input_tensor.shape[-1], kernel_size=(3, 3), padding='same', activation='relu', groups=input_tensor.shape[-1])(input_tensor)

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main path
    block1_output = block_1(input_layer)
    block2_output = block_2(block1_output)
    block3_output = block_3(block2_output)
    block1_repeat_output = block_1(block3_output)

    # Branch path
    branch_output = AveragePooling2D(pool_size=(2, 2), strides=2)(input_layer)

    # Concatenate the main path and the branch path
    combined_output = Concatenate()([block1_repeat_output, branch_output])
    
    # Flatten and fully connected layer
    flatten_output = Flatten()(combined_output)
    dense_output = Dense(units=10, activation='softmax')(flatten_output)  # CIFAR-10 has 10 classes

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=dense_output)

    return model