import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Lambda, BatchNormalization, Flatten, Dense, Add, Reshape, Permute
from tensorflow.keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Split and 1x1 Convolution
    def block1(input_tensor):
        # Split the input into three groups
        splits = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)
        
        # Process each group with a 1x1 convolutional layer
        processed_splits = [Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split) for split in splits]
        
        # Concatenate the processed splits along the channel dimension
        concatenated = Concatenate(axis=-1)(processed_splits)
        
        return concatenated

    # Apply Block 1
    block1_output = block1(input_layer)

    # Block 2: Reshape and Permute
    def block2(input_tensor):
        # Get the shape of the input tensor
        shape = tf.shape(input_tensor)
        
        # Reshape the input tensor into groups
        reshaped = Reshape((shape[1], shape[2], 3, int(shape[3] / 3)))(input_tensor)
        
        # Permute the dimensions to swap the third and fourth dimensions
        permuted = Permute((1, 2, 4, 3))(reshaped)
        
        # Reshape back to the original shape
        shuffled = Reshape((shape[1], shape[2], int(shape[3] / 3) * 3))(permuted)
        
        return shuffled

    # Apply Block 2
    block2_output = block2(block1_output)

    # Block 3: 3x3 Depthwise Separable Convolution
    def block3(input_tensor):
        # Apply 3x3 depthwise separable convolution
        separable_conv = Conv2D(filters=32, kernel_size=(3, 3), depth_multiplier=1, activation='relu', padding='same')(input_tensor)
        
        return separable_conv

    # Apply Block 3
    block3_output = block3(block2_output)

    # Branch to the input
    branch = input_layer

    # Add the main path and the branch outputs
    combined = Add()([block3_output, branch])

    # Flatten the combined output
    flattened = Flatten()(combined)

    # Add fully connected layers
    dense1 = Dense(units=256, activation='relu')(flattened)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model