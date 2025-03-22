# Import necessary packages
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def dl_model():
    """
    This function constructs a deep learning model for image classification using the CIFAR-10 dataset.

    The model begins by defining a basic block, which comprises a convolutional layer, batch normalization, and ReLU activation as its main path.
    The block's branch connects directly to the input. Feature fusion occurs by adding the outputs from both paths.
    An initial convolutional layer reduces the dimensionality of the input features to 16.
    The main structure of the model consists of two consecutive basic blocks, while the branch extracts features via another convolutional layer.
    The outputs from both paths are combined again through addition, enhancing feature representation.
    Finally, an average pooling layer downsamples the feature map, which is then flattened and processed through a fully connected layer to produce the output for the classification task.
    """

    # Define the input shape of the CIFAR-10 dataset
    input_shape = (32, 32, 3)

    # Create an input layer with the specified input shape
    inputs = Input(shape=input_shape)

    # Define an initial convolutional layer with 16 filters, kernel size 3, and ReLU activation
    x = Conv2D(16, 3, activation='relu')(inputs)

    # Define a basic block
    def basic_block(x):
        residual = x
        x = Conv2D(16, 3, activation='relu')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(16, 3, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Add()([x, residual])
        return x

    # Define the main structure of the model
    x = basic_block(x)
    x = basic_block(x)

    # Define the branch
    branch = Conv2D(16, 3, activation='relu')(x)

    # Combine the outputs from both paths through addition
    x = Add()([x, branch])

    # Define another basic block for the branch
    branch = basic_block(branch)

    # Combine the outputs from both paths through addition again
    x = Add()([x, branch])

    # Downsample the feature map using an average pooling layer
    x = AveragePooling2D(2)(x)

    # Flatten the feature map
    x = Flatten()(x)

    # Define a fully connected layer with 10 units for the classification task
    outputs = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs, outputs)

    return model