import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, SeparableConv2D, Concatenate, Reshape, Permute, Flatten, Dense

def dl_model():
    # Block 1
    def block1(inputs):
        # 1x1 convolution
        x1 = Conv2D(32, (1, 1), activation='relu')(inputs)
        
        # 3x3 depthwise separable convolution
        x2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same')(x1)
        x2 = Conv2D(64, (1, 1), activation='relu')(x2)
        
        # Another 1x1 convolution
        x3 = Conv2D(64, (1, 1), activation='relu')(x2)
        
        # Branch path
        x_branch = DepthwiseConv2D(kernel_size=(3, 3), padding='same')(x1)
        x_branch = Conv2D(64, (1, 1), activation='relu')(x_branch)
        
        # Concatenate features from both paths
        x = Concatenate(axis=-1)([x3, x_branch])
        
        return x
    
    # Block 2
    def block2(inputs):
        # Get the shape of the features from Block 1
        shape = tf.keras.backend.int_shape(inputs)
        
        # Reshape into (height, width, groups, channels_per_group)
        x = Reshape((shape[1], shape[2], 4, int(shape[3]/4)))(inputs)
        
        # Swap the third and fourth dimensions
        x = Permute((1, 2, 4, 3))(x)
        
        # Reshape back to the original shape
        x = Reshape(shape)(x)
        
        return x
    
    # Input layer
    inputs = Input(shape=(28, 28, 1))
    
    # Block 1 processing
    x = block1(inputs)
    
    # Block 2 processing
    x = block2(x)
    
    # Flatten and fully connected layer
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()