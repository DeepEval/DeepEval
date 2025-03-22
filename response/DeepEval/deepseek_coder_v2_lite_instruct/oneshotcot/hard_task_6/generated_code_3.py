import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Permute, DepthwiseConv2D, AveragePooling2D
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    def block1(input_tensor):
        # Split the input into three groups
        split1 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)
        split2 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)
        split3 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)
        
        # Process each group with a 1x1 convolutional layer
        processed1 = Conv2D(64, (1, 1), activation='relu')(split1)
        processed2 = Conv2D(64, (1, 1), activation='relu')(split2)
        processed3 = Conv2D(64, (1, 1), activation='relu')(split3)
        
        # Concatenate the processed outputs
        output_tensor = Concatenate()([processed1, processed2, processed3])
        return output_tensor
    
    def block2(input_tensor):
        # Get the shape of the input
        shape = tf.shape(input_tensor)
        height, width, channels = shape[1], shape[2], shape[3]
        
        # Reshape the input to (height, width, groups, channels_per_group)
        reshaped = tf.reshape(input_tensor, (height, width, 3, channels // 3))
        
        # Swap the third and fourth dimensions
        permuted = Permute((1, 2, 4, 3))(reshaped)
        
        # Reshape back to the original shape
        output_tensor = tf.reshape(permuted, (height, width, channels))
        return output_tensor
    
    def block3(input_tensor):
        # Apply a 3x3 depthwise separable convolution
        output_tensor = DepthwiseConv2D((3, 3), padding='same')(input_tensor)
        output_tensor = BatchNormalization()(output_tensor)
        output_tensor = Conv2D(128, (1, 1), activation='relu')(output_tensor)
        return output_tensor
    
    # Apply Block 1
    block1_output = block1(input_layer)
    
    # Apply Block 2
    block2_output = block2(block1_output)
    
    # Apply Block 3
    block3_output = block3(block2_output)
    
    # Repeat Block 1
    block1_repeated = block1(block1_output)
    
    # Concatenate the main path outputs
    main_path_output = Concatenate()([block1_repeated, block3_output])
    
    # Branch path: Average Pooling
    branch_path_output = AveragePooling2D((4, 4))(input_layer)
    branch_path_output = Flatten()(branch_path_output)
    branch_path_output = Dense(128, activation='relu')(branch_path_output)
    
    # Concatenate the main path and branch path outputs
    combined_output = Concatenate()([main_path_output, branch_path_output])
    
    # Final fully connected layer
    output_layer = Dense(10, activation='softmax')(combined_output)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.summary()