import keras
from keras.layers import Input, Conv2D, DepthwiseSeparableConv2D, Flatten, Dense
from keras.models import Model

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the 1x1 convolutional layer
    conv1 = Conv2D(32, (1, 1), activation='relu')

    # Define the 3x3 depthwise separable convolutional layer
    dwconv = DepthwiseSeparableConv2D(32, (3, 3), activation='relu')

    # Define the 1x1 convolutional layer
    conv2 = Conv2D(32, (1, 1), activation='relu')

    # Define the branch 1
    branch1 = conv1(Input(input_shape))
    branch1 = dwconv(branch1)
    branch1 = conv2(branch1)

    # Define the branch 2
    branch2 = conv1(Input(input_shape))
    branch2 = dwconv(branch2)
    branch2 = conv2(branch2)

    # Define the branch 3
    branch3 = conv1(Input(input_shape))
    branch3 = dwconv(branch3)
    branch3 = conv2(branch3)

    # Concatenate the branches
    concatenated = keras.layers.concatenate([branch1, branch2, branch3], axis=1)

    # Flatten the concatenated branches
    flattened = Flatten()(concatenated)

    # Add a fully connected layer
    fc = Dense(10, activation='softmax')(flattened)

    # Define the model
    model = Model(inputs=Input(input_shape), outputs=fc)

    return model