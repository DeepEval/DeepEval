import keras
import tensorflow as tf
from keras.layers import Input, Lambda, DepthwiseConv2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Split the input into three groups
    inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)

    # Apply depthwise separable convolutions to each group
    conv1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(inputs_groups[0])
    conv2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(inputs_groups[1])
    conv3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(inputs_groups[2])

    # Concatenate the outputs of the three convolutions
    concatenated_output = Concatenate()([conv1, conv2, conv3])

    # Flatten the concatenated output
    flatten_output = Flatten()(concatenated_output)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_output)  # CIFAR-10 has 10 classes

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model