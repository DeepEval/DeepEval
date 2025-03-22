import keras
from keras.layers import Input, Lambda, DepthwiseConv2D, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 channels

    # Split the input into three groups along the last dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    def depthwise_separable_conv(input_tensor, kernel_size):
        # Applying Depthwise Separable Convolution
        depthwise_conv = DepthwiseConv2D(kernel_size=kernel_size, padding='same', activation='relu')(input_tensor)
        return depthwise_conv

    # Extract features from each split using different kernel sizes
    path1 = depthwise_separable_conv(split_layer[0], kernel_size=(1, 1))
    path2 = depthwise_separable_conv(split_layer[1], kernel_size=(3, 3))
    path3 = depthwise_separable_conv(split_layer[2], kernel_size=(5, 5))

    # Concatenate the outputs of the three paths
    concatenated = Concatenate()([path1, path2, path3])
    
    # Flatten the concatenated features
    flatten_layer = Flatten()(concatenated)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # CIFAR-10 has 10 classes

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model