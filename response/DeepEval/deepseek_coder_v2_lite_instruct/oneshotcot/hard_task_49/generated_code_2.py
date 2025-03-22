import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense, Lambda, DepthwiseConv2D
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # First block
    def first_block(input_tensor):
        # Pooling layers
        pool1 = AveragePooling2D(pool_size=(1, 1), strides=1)(input_tensor)
        pool2 = AveragePooling2D(pool_size=(2, 2), strides=2)(input_tensor)
        pool3 = AveragePooling2D(pool_size=(4, 4), strides=4)(input_tensor)
        
        # Flatten each pooled output
        flatten1 = Flatten()(pool1)
        flatten2 = Flatten()(pool2)
        flatten3 = Flatten()(pool3)
        
        # Concatenate the flattened outputs
        concatenated = Concatenate()([flatten1, flatten2, flatten3])
        
        return concatenated
    
    # Apply first block
    first_block_output = first_block(input_layer)
    
    # Fully connected layer and reshape to 4D tensor
    fc_layer = Dense(units=64, activation='relu')(first_block_output)
    reshape_layer = Lambda(lambda x: tf.reshape(x, (-1, 2, 2, 16)))(fc_layer)  # 2x2 grid, 16 channels
    
    # Second block
    def second_block(input_tensor):
        # Split the input into four groups
        split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1))(input_tensor)
        
        # Process each group with depthwise separable convolutional layers
        outputs = []
        for i, split in enumerate(split_layer):
            if i == 0:
                conv = DepthwiseConv2D(kernel_size=(1, 1), padding='same')(split)
            elif i == 1:
                conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same')(split)
            elif i == 2:
                conv = DepthwiseConv2D(kernel_size=(5, 5), padding='same')(split)
            elif i == 3:
                conv = DepthwiseConv2D(kernel_size=(7, 7), padding='same')(split)
            
            outputs.append(conv)
        
        # Concatenate the outputs from the four groups
        concatenated = Concatenate()(outputs)
        
        return concatenated
    
    # Apply second block
    second_block_output = second_block(reshape_layer)
    
    # Flatten the final output
    flattened = Flatten()(second_block_output)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flattened)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model