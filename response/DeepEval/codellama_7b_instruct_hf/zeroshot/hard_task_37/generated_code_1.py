from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the parallel branches
    branch1 = Conv2D(32, (3, 3), activation='relu')(input_shape)
    branch2 = Conv2D(64, (3, 3), activation='relu')(input_shape)

    # Define the main paths
    branch1 = Conv2D(32, (3, 3), activation='relu')(branch1)
    branch1 = Conv2D(32, (3, 3), activation='relu')(branch1)
    branch1 = Conv2D(32, (3, 3), activation='relu')(branch1)

    branch2 = Conv2D(64, (3, 3), activation='relu')(branch2)
    branch2 = Conv2D(64, (3, 3), activation='relu')(branch2)
    branch2 = Conv2D(64, (3, 3), activation='relu')(branch2)

    # Define the parallel branch
    parallel_branch = Conv2D(128, (3, 3), activation='relu')(input_shape)

    # Define the addition layer
    addition_layer = Add()([branch1, branch2, parallel_branch])

    # Define the flattening layer
    flattening_layer = Flatten()(addition_layer)

    # Define the fully connected layer
    output_layer = Dense(10, activation='softmax')(flattening_layer)

    # Define the model
    model = Model(inputs=input_shape, outputs=output_layer)

    return model