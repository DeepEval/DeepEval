# Import necessary packages from Keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    """
    This function constructs a deep learning model using Functional APIs of Keras 
    for image classification on the CIFAR-10 dataset.

    The model consists of a main path and a branch path. The main path involves 
    a 1x1 initial convolutional layer followed by three distinct branches, each 
    involving feature extraction, downsampling, upsampling, and concatenation 
    of the outputs from all branches. The branch path also begins with a 1x1 
    convolutional layer to match the number of channels in the main path. The 
    outputs from both paths are added and processed through two fully connected 
    layers for classification across 10 classes.

    Args:
    None

    Returns:
    model: The constructed deep learning model
    """

    # Define the input shape and number of classes
    input_shape = (32, 32, 3)  # CIFAR-10 dataset has 32x32 images with 3 color channels
    num_classes = 10

    # Define the main path
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (1, 1), activation='relu')(inputs)  # Initial 1x1 convolutional layer

    # Branch 1: 3x3 convolutional layer for feature extraction
    branch1 = Conv2D(32, (3, 3), activation='relu')(x)

    # Branch 2: Max pooling, 3x3 convolutional layer, and upsampling
    branch2 = MaxPooling2D((2, 2))(x)
    branch2 = Conv2D(32, (3, 3), activation='relu')(branch2)
    branch2 = UpSampling2D((2, 2))(branch2)

    # Branch 3: Max pooling, 3x3 convolutional layer, and upsampling
    branch3 = MaxPooling2D((2, 2))(x)
    branch3 = Conv2D(32, (3, 3), activation='relu')(branch3)
    branch3 = UpSampling2D((2, 2))(branch3)

    # Concatenate the outputs from all branches
    x = Concatenate()([branch1, branch2, branch3])
    x = Conv2D(32, (1, 1), activation='relu')(x)

    # Define the branch path
    branch_path = Conv2D(32, (1, 1), activation='relu')(x)

    # Add the outputs from both paths
    x = Add()([x, branch_path])

    # Flatten the output
    x = Flatten()(x)

    # Define the fully connected layers for classification
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Define the model
    model = Model(inputs, outputs)

    return model