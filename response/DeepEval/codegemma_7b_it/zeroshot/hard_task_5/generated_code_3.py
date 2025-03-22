import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def residual_block(x, filters, kernel_size=3, padding='same', strides=1):
    """
    Builds a residual block with skip connection.
    """
    # Save the input for skip connection
    shortcut = x

    # Perform initial convolution
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Perform second convolution
    x = layers.Conv2D(filters, kernel_size, strides=1, padding=padding)(x)
    x = layers.BatchNormalization()(x)

    # Add skip connection and activation
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)

    return x

def block_1(x, filters, kernel_size=3, padding='same', strides=1):
    """
    Builds the first block with three branches.
    """
    # Split the input into three groups
    x0 = layers.Lambda(lambda x: tf.split(x, 3, axis=3))(x)

    # Process each branch with a 1x1 convolutional layer
    x1 = layers.Conv2D(filters, 1, strides=strides, padding=padding)(x0[0])
    x2 = layers.Conv2D(filters, 1, strides=strides, padding=padding)(x0[1])
    x3 = layers.Conv2D(filters, 1, strides=strides, padding=padding)(x0[2])

    # Concatenate the outputs from each branch
    x = layers.concatenate([x1, x2, x3], axis=3)
    return x

def block_2(x, filters, kernel_size=3, padding='same', strides=1):
    """
    Builds the second block with channel shuffling.
    """
    # Get the shape of the input
    shape = keras.backend.int_shape(x)

    # Reshape the input into three groups
    x = layers.Reshape((shape[1], shape[2], 3, shape[3] // 3))(x)

    # Permute the third and fourth dimensions
    x = layers.Permute((1, 2, 4, 3))(x)

    # Reshape the feature back to the original shape
    x = layers.Reshape((shape[1], shape[2], shape[3]))(x)
    return x

def block_3(x, filters, kernel_size=3, padding='same', strides=1):
    """
    Builds the third block with a depthwise separable convolution.
    """
    # Perform depthwise separable convolution
    x = layers.DepthwiseConv2D(kernel_size, strides=strides, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Perform pointwise convolution
    x = layers.Conv2D(filters, 1, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    return x

def residual_dense_network():
    """
    Builds the Residual Dense Network model.
    """
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # Block 1
    x = block_1(input_layer, 32)
    x = residual_block(x, 32)
    x = residual_block(x, 32)

    # Block 2
    x = block_2(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 64)

    # Block 3
    x = block_3(x, 128)
    x = residual_block(x, 128)
    x = residual_block(x, 128)

    # Branch
    shortcut = layers.Conv2D(128, 1, strides=2, padding='same')(input_layer)
    shortcut = layers.BatchNormalization()(shortcut)

    # Main path
    main_path = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    main_path = residual_block(main_path, 128)
    main_path = residual_block(main_path, 128)

    # Add branch and main path
    x = layers.add([main_path, shortcut])

    # Output layer
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    output_layer = layers.Dense(10, activation='softmax')(x)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Construct the model
model = residual_dense_network()