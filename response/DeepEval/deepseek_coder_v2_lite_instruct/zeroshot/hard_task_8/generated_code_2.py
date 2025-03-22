import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Reshape, Permute, Flatten, Dense

def dl_model():
    # Block 1
    def block1(inputs):
        # 1x1 convolution
        x = Conv2D(32, (1, 1), activation='relu')(inputs)
        
        # 3x3 depthwise separable convolution
        x = DepthwiseConv2D(kernel_size=(3, 3), activation='relu')(x)
        
        # Another 1x1 convolution
        x = Conv2D(64, (1, 1), activation='relu')(x)
        
        # Branch path
        branch = DepthwiseConv2D(kernel_size=(3, 3), activation='relu')(inputs)
        branch = Conv2D(64, (1, 1), activation='relu')(branch)
        
        # Concatenate along the channel dimension
        x = Concatenate(axis=-1)([x, branch])
        
        return x
    
    # Block 2
    def block2(inputs):
        # Get the shape of the features from Block 1
        shape = tf.keras.backend.int_shape(inputs)
        
        # Reshape the features into (height, width, groups, channels_per_group)
        x = Reshape((shape[1], shape[2], 4, int(shape[3] / 4)))(inputs)
        
        # Swap the third and fourth dimensions
        x = Permute((1, 2, 4, 3))(x)
        
        # Reshape back to the original shape to achieve channel shuffling
        x = Reshape(shape)(x)
        
        return x
    
    # Input layer
    inputs = Input(shape=(28, 28, 1))
    
    # Block 1
    x = block1(inputs)
    
    # Block 2
    x = block2(x)
    
    # Flatten the features
    x = Flatten()(x)
    
    # Fully connected layer for classification
    outputs = Dense(10, activation='softmax')(x)
    
    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Create the model
model = dl_model()