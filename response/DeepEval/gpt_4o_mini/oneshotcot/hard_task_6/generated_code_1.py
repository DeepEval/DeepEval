import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense, Lambda, Reshape
import tensorflow as tf

def block1(input_tensor):
    # Split the input into three groups
    splits = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
    # Process each group with a 1x1 convolution and activation
    processed_splits = [Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), padding='same', activation='relu')(split) for split in splits]
    # Concatenate the processed groups
    output_tensor = Concatenate(axis=-1)(processed_splits)
    return output_tensor

def block2(input_tensor):
    # Obtain the shape of the features from Block 1
    shape = tf.shape(input_tensor)
    # Reshape the tensor into (height, width, groups, channels_per_group)
    reshaped = Reshape((shape[1], shape[2], 3, shape[3] // 3))(input_tensor)
    # Permute the dimensions to shuffle channels
    permuted = tf.transpose(reshaped, perm=(0, 1, 3, 2))  # Swap groups and channels
    # Reshape back to the original shape
    output_tensor = Reshape((shape[1], shape[2], shape[3]))(permuted)
    return output_tensor

def block3(input_tensor):
    # Apply 3x3 depthwise separable convolution
    depthwise_conv = Conv2D(filters=input_tensor.shape[-1], kernel_size=(3, 3), padding='same', activation='relu', groups=input_tensor.shape[-1])(input_tensor)
    return depthwise_conv

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    block1_output = block1(input_layer)
    block2_output = block2(block1_output)
    block3_output = block3(block2_output)
    
    # Repeat Block 1
    block1_repeated_output = block1(block3_output)

    # Branch path with average pooling
    branch_output = AveragePooling2D(pool_size=(2, 2))(input_layer)

    # Concatenate main path and branch path outputs
    concatenated_output = Concatenate()([block1_repeated_output, branch_output])

    # Fully connected layer
    flatten_layer = Flatten()(concatenated_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model