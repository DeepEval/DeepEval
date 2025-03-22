import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, Flatten, Dense, Lambda

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 with 3 channels (RGB)
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Split the input into three groups along the last dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    
    # Feature extraction with depthwise separable convolutions
    def depthwise_conv_block(x, kernel_size):
        x = DepthwiseConv2D(kernel_size=kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x
    
    # Apply depthwise separable convolutions with different kernel sizes
    conv1x1 = depthwise_conv_block(split_layer[0], kernel_size=1)
    conv3x3 = depthwise_conv_block(split_layer[1], kernel_size=3)
    conv5x5 = depthwise_conv_block(split_layer[2], kernel_size=5)
    
    # Concatenate the outputs of the three groups
    concatenated = tf.concat([conv1x1, conv3x3, conv5x5], axis=-1)
    
    # Flatten the concatenated features
    flattened = Flatten()(concatenated)
    
    # Fully connected layer for classification
    outputs = Dense(units=10, activation='softmax')(flattened)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()