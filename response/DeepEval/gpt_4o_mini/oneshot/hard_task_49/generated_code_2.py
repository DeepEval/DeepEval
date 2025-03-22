import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Lambda, Reshape, DepthwiseConv2D
import tensorflow as tf

def dl_model():
    
    # Input layer for MNIST dataset (28x28 grayscale images)
    input_layer = Input(shape=(28, 28, 1))
    
    # First block: three average pooling layers with varying scales
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)
    
    # Flatten the outputs of the pooling layers
    flatten1 = Flatten()(avg_pool1)
    flatten2 = Flatten()(avg_pool2)
    flatten3 = Flatten()(avg_pool3)
    
    # Concatenate the flattened outputs
    concatenated = Concatenate()([flatten1, flatten2, flatten3])
    
    # Fully connected layer and reshape for the second block
    dense1 = Dense(units=128, activation='relu')(concatenated)
    reshaped = Reshape((1, 1, 128))(dense1)  # Reshaping to match dimensions for next block

    # Second block: split and apply depthwise separable convolutions
    split_tensors = Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1))(reshaped)

    # Process each split with a different kernel size depthwise separable convolution
    depthwise_conv1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(split_tensors[0])
    depthwise_conv2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(split_tensors[1])
    depthwise_conv3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(split_tensors[2])
    depthwise_conv4 = DepthwiseConv2D(kernel_size=(7, 7), padding='same', activation='relu')(split_tensors[3])
    
    # Concatenate the outputs of the depthwise separable convolutions
    concat_depthwise = Concatenate()([depthwise_conv1, depthwise_conv2, depthwise_conv3, depthwise_conv4])

    # Flatten the concatenated output and pass through final fully connected layer
    flatten_final = Flatten()(concat_depthwise)
    output_layer = Dense(units=10, activation='softmax')(flatten_final)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model