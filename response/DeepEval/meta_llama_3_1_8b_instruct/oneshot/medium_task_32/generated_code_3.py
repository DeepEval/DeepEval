import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    
    input_layer = keras.Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the last dimension
    split_layer = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Define a function to apply depthwise separable convolution with different kernel sizes
    def separable_conv(input_tensor, kernel_size):
        conv_layer = layers.SeparableConv2D(
            filters=32,
            kernel_size=kernel_size,
            activation='relu'
        )(input_tensor)
        return conv_layer
    
    # Apply depthwise separable convolution with different kernel sizes
    group1_conv = separable_conv(split_layer[0], kernel_size=(1, 1))
    group2_conv = separable_conv(split_layer[1], kernel_size=(3, 3))
    group3_conv = separable_conv(split_layer[2], kernel_size=(5, 5))
    
    # Concatenate the outputs of the three groups
    concat_layer = layers.Concatenate()([group1_conv, group2_conv, group3_conv])
    
    # Apply batch normalization
    batch_norm_layer = layers.BatchNormalization()(concat_layer)
    
    # Flatten the output
    flatten_layer = layers.Flatten()(batch_norm_layer)
    
    # Apply fully connected layer for classification
    output_layer = layers.Dense(
        units=10,
        activation='softmax'
    )(flatten_layer)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model