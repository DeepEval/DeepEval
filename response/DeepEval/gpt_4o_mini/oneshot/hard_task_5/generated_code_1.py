import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Reshape, Permute, Add, DepthwiseConv2D, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Each path has a 1x1 convolution to reduce channels
    path1 = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='relu')(split[0])
    path2 = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='relu')(split[1])
    path3 = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='relu')(split[2])
    
    # Concatenate along the channel dimension
    block1_output = Concatenate(axis=-1)([path1, path2, path3])

    # Block 2
    # Reshape for channel shuffling
    batch_size = tf.shape(block1_output)[0]
    height, width, channels = block1_output.shape[1], block1_output.shape[2], block1_output.shape[3]
    groups = 3
    channels_per_group = channels // groups

    reshaped = Reshape((height, width, groups, channels_per_group))(block1_output)
    permuted = Permute((0, 1, 3, 2, 4))(reshaped)  # Swap last two dimensions
    block2_output = Reshape((height, width, channels))(permuted)  # Reshape back to original shape

    # Block 3
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(block2_output)
    
    # Branch from input
    branch = Conv2D(filters=channels, kernel_size=(1, 1), padding='same')(input_layer)

    # Combine main path and branch
    combined_output = Add()([depthwise_conv, branch])

    # Final layer
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Create the model
model = dl_model()
model.summary()