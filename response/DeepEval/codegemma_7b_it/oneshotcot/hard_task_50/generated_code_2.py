import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dense, Dropout, Concatenate, Lambda, Reshape
from keras.initializers import VarianceScaling
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10

def block(input_tensor, kernel_size, pool_size, strides):
    """
    A simple block consisting of a max pooling layer, followed by a flatten operation.

    Args:
        input_tensor: The input tensor to the block.
        kernel_size: The kernel size for the max pooling layer.
        pool_size: The pool size for the max pooling layer.
        strides: The strides for the max pooling layer.

    Returns:
        A flattened tensor from the block.
    """
    x = MaxPooling2D(pool_size=pool_size, strides=strides, padding='same')(input_tensor)
    x = Flatten()(x)
    return x

def separable_conv(input_tensor, kernel_size, filters):
    """
    A separable convolutional layer, consisting of depthwise and pointwise convolutions.

    Args:
        input_tensor: The input tensor to the layer.
        kernel_size: The kernel size for the depthwise convolution.
        filters: The number of filters for the pointwise convolution.

    Returns:
        A tensor after applying the separable convolution.
    """
    x = DepthwiseConv2D(kernel_size, padding='same', use_bias=False)(input_tensor)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)
    x = PointwiseConv2D(filters, (1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)
    return x

def dl_model():
    """
    Constructs the CIFAR-10 image classification model using Functional APIs.

    Returns:
        A Keras model for image classification.
    """
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First block
    x = block(input_layer, kernel_size=1, pool_size=(1, 1), strides=(1, 1))
    y = block(input_layer, kernel_size=2, pool_size=(2, 2), strides=(2, 2))
    z = block(input_layer, kernel_size=4, pool_size=(4, 4), strides=(4, 4))

    # Dropout before concatenation
    x = Dropout(rate=0.2)(x)
    y = Dropout(rate=0.2)(y)
    z = Dropout(rate=0.2)(z)

    # Concatenate flattened outputs from each block
    concat_tensor = Concatenate()([x, y, z])

    # Second block
    groups = 4
    split_tensor = Lambda(lambda x: tf.split(x, num_or_size_splits=groups, axis=-1))(concat_tensor)

    # Separable convolutions for each group
    output_tensors = []
    for i in range(groups):
        output_tensors.append(separable_conv(split_tensor[i], kernel_size=(i+1, i+1), filters=256))

    # Concatenate outputs from all groups
    concat_output = Concatenate()(output_tensors)

    # Reshape for fully connected layer
    reshape_tensor = Reshape((-1,))(concat_output)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(reshape_tensor)

    # Model definition
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    optimizer = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model