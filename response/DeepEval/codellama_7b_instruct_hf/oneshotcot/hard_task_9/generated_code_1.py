from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense
from keras.models import Model
from keras.applications import VGG16


def dl_model():

    # Load the CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Define the input layer with a shape of (32, 32, 3)
    input_layer = Input(shape=(32, 32, 3))

    # Define the first feature extraction branch with a 1x1 convolution
    branch1 = Conv2D(32, (1, 1), activation='relu')(input_layer)

    # Define the second feature extraction branch with a 1x1 convolution followed by a 3x3 convolution
    branch2 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    branch2 = Conv2D(32, (3, 3), activation='relu')(branch2)

    # Define the third feature extraction branch with a 1x1 convolution followed by two 3x3 convolutions
    branch3 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    branch3 = Conv2D(32, (3, 3), activation='relu')(branch3)
    branch3 = Conv2D(32, (3, 3), activation='relu')(branch3)

    # Concatenate the outputs from the three feature extraction branches
    concat = Concatenate()([branch1, branch2, branch3])

    # Apply a 1x1 convolution to adjust the output dimensions to match the input image's channel size
    concat = Conv2D(32, (1, 1), activation='relu')(concat)

    # Define the main path by adding the feature extraction branches and the concatenation layer to the input
    main_path = Concatenate()([input_layer, concat])

    # Define the branch directly connects to the input, and the main path and the branch are fused together through addition
    branch = Conv2D(32, (1, 1), activation='relu')(input_layer)
    main_path = Add()([main_path, branch])

    # Define the final classification result by passing the output of the main path through three fully connected layers
    main_path = Flatten()(main_path)
    main_path = Dense(128, activation='relu')(main_path)
    main_path = Dense(64, activation='relu')(main_path)
    main_path = Dense(10, activation='softmax')(main_path)

    # Return the constructed model
    model = Model(inputs=input_layer, outputs=main_path)

    return model