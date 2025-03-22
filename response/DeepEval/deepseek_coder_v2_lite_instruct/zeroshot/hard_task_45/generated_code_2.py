import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, MaxPooling2D, Concatenate, Flatten, Dense, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    # Split the input into three groups along the last dimension
    split_layers = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)

    # Apply depthwise separable convolutions with different kernel sizes
    conv1x1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(split_layers[0])
    conv3x3 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(split_layers[1])
    conv5x5 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(split_layers[2])

    # Concatenate the outputs
    concat_first_block = Concatenate(axis=-1)([conv1x1, conv3x3, conv5x5])

    # Second block
    # Define multiple branches
    branch1 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    branch2 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    branch3 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    branch4 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    branch5 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    branch6 = Conv2D(32, (3, 3), activation='relu')(input_layer)

    # Concatenate outputs from all branches
    concat_second_block = Concatenate(axis=-1)([branch1, branch2, branch3, branch4, branch5, branch6])

    # Flatten the concatenated output
    flatten_layer = Flatten()(concat_second_block)

    # Fully connected layer
    output_layer = Dense(10, activation='softmax')(flatten_layer)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model