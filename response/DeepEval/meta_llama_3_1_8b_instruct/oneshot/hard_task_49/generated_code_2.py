import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Lambda, Reshape, Dense
from keras.layers import Conv2D, DepthwiseConv2D
from tensorflow.keras import backend as K

def dl_model():
    """
    Construct a deep learning model for image classification using the MNIST dataset.
    The model is structured into two blocks. The first block processes the input through
    three average pooling layers with varying scales, utilizing pooling windows and strides
    of 1x1, 2x2, and 4x4. Each pooling result is flattened into a one-dimensional vector,
    and these vectors are concatenated. Between the first and second blocks, a fully connected
    layer and a reshape operation transform the output of the first block into a 4-dimensional
    tensor, suitable for input into the second block. The second block splits the input into
    four groups along the last dimension by encapsulating tf.split within Lambda layer, each
    processed by depthwise separable convolutional layers with different kernel sizes (1x1, 3x3,
    5x5, and 7x7) for feature extraction. The outputs from these groups are then concatenated.
    Finally, the processed data is flattened and passed through a fully connected layer to produce
    the classification result.
    """

    input_layer = Input(shape=(28, 28, 1))

    # First block: average pooling with varying scales
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(input_layer)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='same')(input_layer)

    # Flatten each pooling result
    flatten1 = Flatten()(pool1)
    flatten2 = Flatten()(pool2)
    flatten3 = Flatten()(pool3)

    # Concatenate the flattened results
    concat = Concatenate()([flatten1, flatten2, flatten3])

    # Fully connected layer and reshape operation
    dense1 = Dense(units=128, activation='relu')(concat)
    reshape = Reshape((4, 128))(dense1)

    # Second block: depthwise separable convolutional layers
    def depthwise_separable_convolution(input_tensor, kernel_size):
        x = DepthwiseConv2D(kernel_size=kernel_size, strides=1, padding='same')(input_tensor)
        x = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same')(x)
        return x

    # Split the input into four groups
    def split(input_tensor):
        return Lambda(lambda x: K.split(x, num_or_size_splits=4, axis=-1))(input_tensor)

    # Process each group with depthwise separable convolutional layers
    group1 = depthwise_separable_convolution(reshape, kernel_size=(1, 1))
    group2 = depthwise_separable_convolution(reshape, kernel_size=(3, 3))
    group3 = depthwise_separable_convolution(reshape, kernel_size=(5, 5))
    group4 = depthwise_separable_convolution(reshape, kernel_size=(7, 7))

    # Concatenate the processed groups
    concat_groups = Concatenate()([group1, group2, group3, group4])

    # Flatten and pass through a fully connected layer
    flatten = Flatten()(concat_groups)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model