# Import necessary packages
from tensorflow.keras.layers import Input, Lambda, DepthwiseConv2D, Conv2D, concatenate, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

def dl_model():
    # Define the input shape of the images in CIFAR-10
    input_shape = (32, 32, 3)

    # Define the input layer
    inputs = Input(shape=input_shape)

    # Split the input into three groups along the last dimension using Lambda layer
    split_layer = Lambda(lambda x: K.split(x, 3, axis=-1))(inputs)

    # Define the feature extraction layers for each group
    group1 = DepthwiseConv2D(kernel_size=(1, 1), activation='relu')(split_layer[0])
    group1 = Conv2D(32, (1, 1), activation='relu')(group1)

    group2 = DepthwiseConv2D(kernel_size=(3, 3), activation='relu')(split_layer[1])
    group2 = Conv2D(32, (3, 3), activation='relu')(group2)

    group3 = DepthwiseConv2D(kernel_size=(5, 5), activation='relu')(split_layer[2])
    group3 = Conv2D(32, (5, 5), activation='relu')(group3)

    # Concatenate the outputs of the three groups
    merged = concatenate([group1, group2, group3], axis=-1)

    # Flatten the merged features into a one-dimensional vector
    flattened = Flatten()(merged)

    # Define the fully connected layer for classification
    outputs = Dense(10, activation='softmax')(flattened)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model