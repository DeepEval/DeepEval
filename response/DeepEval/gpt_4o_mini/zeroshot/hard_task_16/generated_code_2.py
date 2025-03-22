import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Lambda, GlobalMaxPooling2D, Dense, Reshape, Add, Concatenate
from tensorflow.keras.models import Model

def split_and_convolve(x):
    # Split the input into 3 groups along the last dimension
    group1, group2, group3 = tf.split(x, num_or_size_splits=3, axis=-1)
    
    # Define the convolutional blocks
    def conv_block(input_tensor):
        x = Conv2D(32, (1, 1), padding='same', activation='relu')(input_tensor)
        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
        return x
    
    # Apply the convolutional blocks
    conv1 = conv_block(group1)
    conv2 = conv_block(group2)
    conv3 = conv_block(group3)
    
    # Concatenate the outputs
    return Concatenate(axis=-1)([conv1, conv2, conv3])

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 image shape
    inputs = Input(shape=input_shape)
    
    # Block 1
    block1_output = Lambda(split_and_convolve)(inputs)
    
    # Transition Convolution
    transition_conv = Conv2D(64, (1, 1), padding='same')(block1_output)
    
    # Block 2
    global_pool = GlobalMaxPooling2D()(transition_conv)
    
    # Fully connected layers for channel-matching weights
    fc1 = Dense(128, activation='relu')(global_pool)
    fc2 = Dense(transition_conv.shape[-1], activation='sigmoid')(fc1)  # Weights the same size as channels
    
    # Reshape weights and multiply with adjusted output
    reshaped_weights = Reshape((1, 1, transition_conv.shape[-1]))(fc2)  # Reshape for broadcasting
    main_path_output = tf.multiply(transition_conv, reshaped_weights)
    
    # Branch connection
    branch_output = inputs  # Direct connection from input
    
    # Combine main path and branch
    combined_output = Add()([main_path_output, branch_output])
    
    # Final classification layer
    flatten_output = GlobalMaxPooling2D()(combined_output)
    final_output = Dense(10, activation='softmax')(flatten_output)  # CIFAR-10 has 10 classes
    
    # Create the model
    model = Model(inputs=inputs, outputs=final_output)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()  # Uncomment to see the model summary