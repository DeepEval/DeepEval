import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, SeparableConv2D, Concatenate, Reshape, Permute, Dense

def dl_model():
    # Block 1
    def block1(inputs):
        # 1x1 convolution
        x = Conv2D(64, (1, 1), activation='relu')(inputs)
        
        # 3x3 depthwise separable convolution
        x = DepthwiseConv2D(kernel_size=(3, 3), activation='relu')(x)
        
        # Another 1x1 convolution
        x = Conv2D(128, (1, 1), activation='relu')(x)
        
        # Branch path
        branch = DepthwiseConv2D(kernel_size=(3, 3), activation='relu')(inputs)
        branch = Conv2D(128, (1, 1), activation='relu')(branch)
        
        # Concatenate features from both paths
        x = Concatenate(axis=-1)([x, branch])
        
        return x

    # Block 2
    def block2(inputs):
        # Get the shape of the features from Block 1
        shape = tf.keras.backend.int_shape(inputs)
        
        # Reshape the features into (height, width, groups, channels_per_group)
        x = Reshape((shape[1], shape[2], 2, shape[3] // 2))(inputs)
        
        # Swap the third and fourth dimensions
        x = Permute((1, 2, 4, 3))(x)
        
        # Reshape back to the original shape
        x = Reshape((shape[1], shape[2], shape[3] // 2 * 2))(x)
        
        return x

    # Input layer
    inputs = Input(shape=(28, 28, 1))
    
    # Block 1 operations
    x = block1(inputs)
    
    # Block 2 operations
    x = block2(x)
    
    # Flatten the features
    x = tf.keras.layers.Flatten()(x)
    
    # Fully connected layer for classification
    outputs = Dense(10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Create the model
model = dl_model()

# Print the model summary
model.summary()