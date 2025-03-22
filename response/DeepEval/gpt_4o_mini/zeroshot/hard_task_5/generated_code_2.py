import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Lambda, Concatenate, Reshape, Permute, DepthwiseConv2D, Add, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Block 1
    split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    conv1 = Conv2D(filters=1, kernel_size=(1, 1), activation='relu')(split_inputs[0])
    conv2 = Conv2D(filters=1, kernel_size=(1, 1), activation='relu')(split_inputs[1])
    conv3 = Conv2D(filters=1, kernel_size=(1, 1), activation='relu')(split_inputs[2])
    
    # Concatenating the outputs from the three convolutions
    block1_output = Concatenate(axis=-1)([conv1, conv2, conv3])
    
    # Block 2
    shape = tf.shape(block1_output)
    reshaped = Reshape((shape[1], shape[2], 3, shape[3] // 3))(block1_output)  # Reshape to (height, width, groups, channels_per_group)
    permuted = Permute((0, 1, 3, 2))(reshaped)  # Swap the last two dimensions
    block2_output = Reshape((shape[1], shape[2], shape[3]))(permuted)  # Reshape back to original

    # Block 3
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(block2_output)
    
    # Branch connecting directly to the input
    branch_output = inputs  # This is the direct connection from input

    # Combining the outputs from the main path (depthwise_conv) and the branch
    combined_output = Add()([depthwise_conv, branch_output])

    # Final classification layer
    final_output = GlobalAveragePooling2D()(combined_output)
    final_output = Dense(units=10, activation='softmax')(final_output)  # CIFAR-10 has 10 classes

    # Creating the model
    model = Model(inputs=inputs, outputs=final_output)

    return model

# Example of creating the model
model = dl_model()
model.summary()