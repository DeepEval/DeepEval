# Import necessary packages
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    # Define the input shape of the CIFAR-10 dataset
    input_shape = (32, 32, 3)

    # Define the input layer
    inputs = Input(shape=input_shape)

    # First branch: dimensionality reduction using 1x1 convolution
    branch1 = Conv2D(32, (1, 1), activation='relu')(inputs)
    
    # Second branch: extracting features using 1x1 and 3x3 convolution
    branch2 = Conv2D(32, (1, 1), activation='relu')(inputs)
    branch2 = Conv2D(32, (3, 3), activation='relu')(branch2)
    
    # Third branch: capturing larger spatial information using 1x1 and 5x5 convolution
    branch3 = Conv2D(32, (1, 1), activation='relu')(inputs)
    branch3 = Conv2D(32, (5, 5), activation='relu')(branch3)
    
    # Fourth branch: downsampling using 3x3 max pooling followed by 1x1 convolution
    branch4 = MaxPooling2D((3, 3))(inputs)
    branch4 = Conv2D(32, (1, 1), activation='relu')(branch4)

    # Concatenate the outputs of the branches
    concatenated = concatenate([branch1, branch2, branch3, branch4], axis=-1)

    # Flatten the concatenated features
    flattened = Flatten()(concatenated)

    # Add two fully connected layers for classification
    x = Dense(64, activation='relu')(flattened)
    outputs = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs, outputs)

    return model