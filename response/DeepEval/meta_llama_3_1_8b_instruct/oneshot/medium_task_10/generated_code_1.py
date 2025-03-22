import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, AveragePooling2D, Flatten, Dense

def dl_model():     
    """
    Define a deep learning model for image classification using the CIFAR-10 dataset.

    The model begins by adjusting the input feature dimensionality to 16 using a convolutional layer.
    It employs a basic block where the main path includes convolution, batch normalization, and ReLU activation,
    while the branch connects directly to the block's input. The outputs from both paths are combined through an
    addition operation. The core architecture of the model utilizes these basic blocks to create a three-level
    residual connection structure.

    Finally, average pooling followed by a fully connected layer maps the features to classification probabilities
    across 10 classes, making it suitable for multi-class classification tasks.

    Returns:
        model: The constructed Keras model.
    """

    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Adjust the input feature dimensionality to 16
    conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # First level: a basic block
    block1 = basic_block(conv)
    block1 = basic_block(block1)

    # Second level: two residual blocks
    block2 = basic_block(block1)
    block2 = basic_block(block2)

    # Third level: global branch and residual connection
    block3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    block3 = basic_block(block3)
    block2 = Add()([block2, block3])

    # Average pooling
    avg_pool = AveragePooling2D(pool_size=(8, 8))(block2)

    # Flatten the output
    flatten_layer = Flatten()(avg_pool)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Define the Keras model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

def basic_block(input_tensor):
    """
    Define a basic block for the residual connection structure.

    Args:
        input_tensor: The input tensor.

    Returns:
        The output tensor after the basic block.
    """

    # Main path: convolution, batch normalization, and ReLU activation
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
    main_path = BatchNormalization()(main_path)
    main_path = Activation('relu')(main_path)

    # Branch: convolution
    branch = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)

    # Combine the outputs from both paths
    output_tensor = Add()([main_path, branch])

    # Apply ReLU activation
    output_tensor = Activation('relu')(output_tensor)

    return output_tensor