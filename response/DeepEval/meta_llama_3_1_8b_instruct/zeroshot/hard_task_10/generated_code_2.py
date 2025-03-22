from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, Concatenate
from keras.layers import Reshape, Conv2DTranspose, ZeroPadding2D
from keras.models import Model

def dl_model():
    # Define the input shape of the images (32x32x3 for CIFAR-10)
    input_shape = (32, 32, 3)

    # Define the input layer
    inputs = Input(shape=input_shape)

    # Feature extraction path 1: 1x1 convolution
    path1 = Conv2D(32, (1, 1), activation='relu')(inputs)
    path1 = MaxPooling2D((2, 2))(path1)
    path1 = Conv2D(64, (3, 3), activation='relu')(path1)
    path1 = MaxPooling2D((2, 2))(path1)
    path1 = Conv2D(128, (3, 3), activation='relu')(path1)

    # Feature extraction path 2: 1x7 convolution, followed by 1x7 and then 7x1
    path2 = Conv2D(32, (1, 1), activation='relu')(inputs)
    path2 = ZeroPadding2D(((3, 3), (0, 0)))(path2)
    path2 = Conv2D(32, (7, 1), activation='relu')(path2)
    path2 = ZeroPadding2D(((0, 0), (3, 3)))(path2)
    path2 = Conv2D(32, (1, 7), activation='relu')(path2)
    path2 = MaxPooling2D((2, 2))(path2)
    path2 = Conv2D(64, (3, 3), activation='relu')(path2)
    path2 = MaxPooling2D((2, 2))(path2)
    path2 = Conv2D(128, (3, 3), activation='relu')(path2)

    # Concatenate the outputs from the two paths
    concatenated = Concatenate()([path1, path2])

    # Align the output dimensions with the input image's channel
    output = Conv2D(128, (1, 1), activation='relu')(concatenated)

    # Direct branch from the input
    branch = inputs

    # Merge the outputs of the main path and the branch through addition
    merged = Add()([output, branch])

    # Flatten the merged output
    flat = Flatten()(merged)

    # Classification results through two fully connected layers
    outputs = Dense(64, activation='relu')(flat)
    outputs = Dense(10, activation='softmax')(outputs)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Test the model
model = dl_model()
model.summary()