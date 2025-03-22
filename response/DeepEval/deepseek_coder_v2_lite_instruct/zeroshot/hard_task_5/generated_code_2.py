import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense, Add, Reshape, Permute

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Block 1
    # Split the input into three groups
    split_1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(inputs)
    
    # Process each group with a 1x1 convolutional layer
    conv_blocks = []
    for group in split_1:
        conv_block = Conv2D(filters=group.shape[3] // 3, kernel_size=(1, 1), padding='same')(group)
        conv_blocks.append(conv_block)
    
    # Concatenate the outputs from the three groups
    fused_features = tf.concat(conv_blocks, axis=3)
    
    # Block 2
    # Get the shape of the fused features
    shape = fused_features.shape
    height, width, channels = shape[1], shape[2], shape[3]
    
    # Reshape into (height, width, groups, channels_per_group)
    reshaped_features = Reshape((height, width, 3, channels // 3))(fused_features)
    
    # Swap the third and fourth dimensions
    permuted_features = Permute((1, 2, 4, 3))(reshaped_features)
    
    # Reshape back to the original shape to achieve channel shuffling
    shuffled_features = Reshape(shape[1:])(permuted_features)
    
    # Block 3
    # Process with a 3x3 depthwise separable convolution
    separable_conv = Conv2D(filters=channels, kernel_size=(3, 3), padding='same', depthwise_mode=True)(shuffled_features)
    
    # Branch to the input
    branch = inputs
    
    # Add the outputs from the main path and the branch
    added_features = Add()([separable_conv, branch])
    
    # Flatten the output
    flattened_features = Flatten()(added_features)
    
    # Pass through a fully connected layer for classification
    outputs = Dense(units=10, activation='softmax')(flattened_features)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()