import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Permute, DepthwiseConv2D
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three groups
    group_indices = tf.split(input_layer, 3, axis=2)
    
    # Process each group with a 1x1 convolutional layer
    conv1 = Conv2D(filters=1/3, kernel_size=(1, 1), activation='relu')(group_indices[0])
    conv2 = Conv2D(filters=1/3, kernel_size=(1, 1), activation='relu')(group_indices[1])
    conv3 = Conv2D(filters=1/3, kernel_size=(1, 1), activation='relu')(group_indices[2])
    
    # Concatenate the outputs from the three groups
    fused_features = Concatenate()(conv1, conv2, conv3)
    
    # Block 1 (Feature Transformation)
    block1_output = fused_features
    
    # Block 2 (Channel Shuffle and Depthwise Convolution)
    block2_output = block1_output
    
    # Obtain the shape of the feature from Block 1
    shape_block1 = tf.shape(block1_output)
    
    # Reshape into three groups with target shape
    group1 = Permute((2, 3, 1))(block1_output)
    group2 = Permute((2, 3, 1))(block1_output)
    group3 = Permute((2, 3, 1))(block1_output)
    
    # Reshape back to original shape
    reshaped_group1 = Permute((1, 2, 3))(group1)
    reshaped_group2 = Permute((1, 2, 3))(group2)
    reshaped_group3 = Permute((1, 2, 3))(group3)
    
    # Apply 3x3 depthwise separable convolution
    dw_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1)(reshaped_group3)
    
    # Branch connection to the input
    branch_output = Conv2D(filters=3, kernel_size=(1, 1), activation='sigmoid')(input_layer)
    
    # Combine outputs from the main path and the branch
    combined_output = Concatenate()([dw_conv, reshaped_group1, reshaped_group2, reshaped_group3, branch_output])
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(combined_output)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.summary()