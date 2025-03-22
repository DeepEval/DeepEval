import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Reshape, Permute
from tensorflow.keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 2: Add convolutional layer (Initial convolutional layer with 32 kernels)
    conv_initial = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 3: Split the input into two groups
    split_layer = Lambda(lambda x: tf.split(x, 2, axis=-1))(conv_initial)
    
    # Block 1
    def block1(input_tensor):
        # First group operations
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_mode=True)(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)
        
        # Second group (pass through without modification)
        path4 = input_tensor
        
        # Merge outputs
        output_tensor = Concatenate()([path1, path3, path4])
        return output_tensor
    
    # Apply Block 1
    block1_output = block1(split_layer[0])
    
    # Block 2
    def block2(input_tensor):
        # Get shape of input
        input_shape = tf.shape(input_tensor)
        height = input_shape[1]
        width = input_shape[2]
        groups = 4
        channels_per_group = input_shape[-1] // groups
        
        # Reshape into groups
        reshaped = Reshape(target_shape=(height, width, groups, channels_per_group))(input_tensor)
        
        # Swap third and fourth dimensions
        permuted = Permute((1, 2, 4, 3))(reshaped)
        
        # Flatten the result
        flattened = Flatten()(permuted)
        
        return flattened
    
    # Apply Block 2
    block2_output = block2(block1_output)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(block2_output)
    
    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model