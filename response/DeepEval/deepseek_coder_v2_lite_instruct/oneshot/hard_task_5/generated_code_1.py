import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Permute, DepthwiseConv2D
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    def block1(input_tensor):
        # Split the input into three groups
        split1, split2, split3 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Process each split with a 1x1 convolutional layer
        processed1 = Conv2D(filters=int(split1.shape[-1]/3), kernel_size=(1, 1), activation='relu')(split1)
        processed2 = Conv2D(filters=int(split2.shape[-1]/3), kernel_size=(1, 1), activation='relu')(split2)
        processed3 = Conv2D(filters=int(split3.shape[-1]/3), kernel_size=(1, 1), activation='relu')(split3)
        
        # Concatenate the processed outputs
        concatenated = Concatenate(axis=-1)([processed1, processed2, processed3])
        return concatenated
    
    def block2(input_tensor):
        # Get the shape of the input tensor
        shape = input_tensor.shape
        
        # Reshape into groups
        reshaped = Lambda(lambda x: tf.reshape(x, (shape[1], shape[2], 3, int(shape[-1]/3))))(input_tensor)
        
        # Permute dimensions to swap the third and fourth dimensions
        permuted = Permute((1, 2, 4, 3))(reshaped)
        
        # Reshape back to original shape for channel shuffling
        shuffled = Lambda(lambda x: tf.reshape(x, (shape[1], shape[2], shape[-1])))(permuted)
        return shuffled
    
    def block3(input_tensor):
        # Apply 3x3 depthwise separable convolution
        depthwise = DepthwiseConv2D(kernel_size=(3, 3), padding='same')(input_tensor)
        return depthwise
    
    # Apply Block 1
    block1_output = block1(input_layer)
    
    # Apply Block 2
    block2_output = block2(block1_output)
    
    # Apply Block 3
    block3_output = block3(block2_output)
    
    # Add the output of Block 3 to the input (branch)
    added = tf.add(block3_output, input_layer)
    
    # Flatten the result
    flattened = Flatten()(added)
    
    # Pass through fully connected layers
    dense1 = Dense(256, activation='relu')(flattened)
    dense2 = Dense(128, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.summary()