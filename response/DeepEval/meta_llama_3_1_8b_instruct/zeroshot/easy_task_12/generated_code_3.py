# Import necessary packages
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import activations
from tensorflow.keras import regularizers

def dl_model():
    """
    Creates a deep learning model for image classification using the MNIST dataset.
    
    The model architecture includes both a main path and a branch path. The main path consists of two blocks,
    each featuring a ReLU and a separable convolutional layer, followed by a max pooling layer. The branch path
    employs a 1x1 convolutional layer to match the output dimensions of the main and branch paths. Finally, 
    the outputs from both paths are summed, followed by a flattening layer and a fully connected layer to generate 
    the final probability distribution.
    
    Returns:
        A Keras model instance.
    """

    # Define the main path
    main_path = layers.Input(shape=(28, 28, 1), name='main_input')
    x = layers.ReLU()(main_path)
    x = layers.SeparableConv2D(32, (3, 3), activation='relu', name='main_separable_conv_1')(x)
    x = layers.MaxPooling2D((2, 2), name='main_max_pool_1')(x)
    x = layers.ReLU()(x)
    x = layers.SeparableConv2D(64, (3, 3), activation='relu', name='main_separable_conv_2')(x)
    x = layers.MaxPooling2D((2, 2), name='main_max_pool_2')(x)

    # Define the branch path
    branch_path = layers.Input(shape=(28, 28, 1), name='branch_input')
    x_branch = layers.ReLU()(branch_path)
    x_branch = layers.Conv2D(64, (1, 1), activation='relu', name='branch_conv')(x_branch)
    x_branch = layers.MaxPooling2D((2, 2), name='branch_max_pool')(x_branch)

    # Sum the outputs from both paths and add a 1x1 convolutional layer
    x = layers.Concatenate()([x, x_branch])
    x = layers.Conv2D(64, (1, 1), activation='relu', name='sum_conv')(x)

    # Flatten the output and add a fully connected layer
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01), name='dense_1')(x)
    x = layers.Dense(10, activation='softmax', name='output')(x)

    # Create the model
    model = Model(inputs=[main_path, branch_path], outputs=x)

    return model