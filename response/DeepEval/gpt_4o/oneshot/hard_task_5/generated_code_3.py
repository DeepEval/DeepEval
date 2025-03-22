import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Add, Dense, DepthwiseConv2D, GlobalAveragePooling2D
import tensorflow as tf

def block1(input_tensor):
    # Split the input into 3 groups
    channels = input_tensor.shape[-1]
    split_channels = channels // 3

    groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)

    # Apply 1x1 convolution to each group
    conv_groups = [Conv2D(filters=split_channels, kernel_size=(1, 1), activation='relu')(group) for group in groups]

    # Concatenate the groups back together
    output_tensor = Concatenate(axis=-1)(conv_groups)

    return output_tensor

def block2(input_tensor):
    # Reshape to (height, width, groups, channels_per_group)
    height, width, channels = input_tensor.shape[1:]
    groups = 3
    channels_per_group = channels // groups

    reshaped = Lambda(lambda x: tf.reshape(x, [-1, height, width, groups, channels_per_group]))(input_tensor)

    # Swap the third and fourth dimensions
    shuffled = Lambda(lambda x: tf.transpose(x, [0, 1, 2, 4, 3]))(reshaped)

    # Reshape back to original shape
    output_tensor = Lambda(lambda x: tf.reshape(x, [-1, height, width, channels]))(shuffled)

    return output_tensor

def block3(input_tensor):
    # Apply 3x3 depthwise separable convolution
    output_tensor = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
    
    return output_tensor

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    x = block1(input_layer)
    x = block2(x)
    x = block3(x)
    x = block1(x)  # Re-using Block 1 as specified

    # Branch that connects directly to the input
    branch = input_layer

    # Combine main path and branch
    combined = Add()([x, branch])

    # Global average pooling to reduce the spatial dimensions
    pooled = GlobalAveragePooling2D()(combined)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(pooled)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model