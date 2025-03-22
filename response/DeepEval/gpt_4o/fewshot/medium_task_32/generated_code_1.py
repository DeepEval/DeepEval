import keras
import tensorflow as tf
from keras.layers import Input, Lambda, DepthwiseConv2D, Concatenate, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Split the input into three groups along the last dimension
    input_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)

    # Apply depthwise separable convolutional layers with different kernel sizes
    conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_groups[0])
    conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_groups[1])
    conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_groups[2])

    # Concatenate the outputs of the three convolutional layers
    concatenated = Concatenate()([conv1, conv2, conv3])

    # Flatten the concatenated features into a one-dimensional vector
    flatten = Flatten()(concatenated)

    # Pass through a fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model