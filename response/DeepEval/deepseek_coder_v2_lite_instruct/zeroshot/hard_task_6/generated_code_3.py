import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Concatenate, Lambda, Permute, DepthwiseConv2D

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)
    
    # Define the inputs
    inputs = Input(shape=input_shape)
    
    # Block 1
    def block1(x):
        # Split the input into three groups
        splits = tf.split(x, num_or_size_splits=3, axis=-1)
        
        # Process each group with a 1x1 convolutional layer followed by an activation function
        processed_groups = [Conv2D(filters=int(x.shape[-1] / 3), kernel_size=(1, 1), activation='relu')(split) for split in splits]
        
        # Concatenate the processed groups
        output = Concatenate(axis=-1)(processed_groups)
        return output
    
    x = block1(inputs)
    
    # Block 2
    def block2(x):
        # Get the shape of the features from Block 1
        height, width, channels = x.shape[1], x.shape[2], x.shape[3]
        
        # Reshape the feature into (height, width, groups, channels_per_group)
        x = tf.reshape(x, shape=(-1, height, width, 3, int(channels / 3)))
        
        # Swap the third and fourth dimensions
        x = Permute((1, 2, 4, 3))(x)
        
        # Reshape back to the original shape to achieve channel shuffling
        x = tf.reshape(x, shape=(-1, height, width, channels))
        return x
    
    x = block2(x)
    
    # Block 3
    x = DepthwiseConv2D(kernel_size=(3, 3), activation='relu')(x)
    
    # Branch path: Average Pooling
    branch = AveragePooling2D(pool_size=(8, 8))(inputs)
    branch = Flatten()(branch)
    
    # Concatenate the outputs from the main path and the branch path
    combined = Concatenate(axis=-1)([x, branch])
    
    # Fully connected layer
    outputs = Dense(units=10, activation='softmax')(combined)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()