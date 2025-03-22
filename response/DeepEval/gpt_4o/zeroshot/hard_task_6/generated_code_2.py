import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Lambda, AveragePooling2D, Concatenate, DepthwiseConv2D
from tensorflow.keras.models import Model

def block1(x):
    # Split input into three groups
    split_channels = tf.keras.layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(x)
    
    # Process each group with a 1x1 convolution and activation function
    convs = [Conv2D(filters=x.shape[-1] // 3, kernel_size=(1, 1), activation='relu')(group) for group in split_channels]
    
    # Concatenate outputs
    return Concatenate()(convs)

def block2(x):
    # Obtain shape and reshape
    height, width, channels = x.shape[1], x.shape[2], x.shape[3]
    x = tf.keras.layers.Reshape((height, width, 3, channels // 3))(x)
    
    # Permute dimensions
    x = tf.keras.layers.Permute((1, 2, 4, 3))(x)
    
    # Reshape back to original shape
    x = tf.keras.layers.Reshape((height, width, channels))(x)
    
    return x

def block3(x):
    # Apply 3x3 depthwise separable convolution
    return DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(x)

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Main path
    x = block1(inputs)
    x = block2(x)
    x = block3(x)
    x = block1(x)  # Repeated Block 1

    # Branch path
    branch = AveragePooling2D(pool_size=(32, 32))(inputs)
    branch = Flatten()(branch)
    
    # Concatenate main path and branch path
    x = Flatten()(x)
    concatenated = Concatenate()([x, branch])
    
    # Fully connected layer for classification
    outputs = Dense(10, activation='softmax')(concatenated)
    
    # Construct model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()