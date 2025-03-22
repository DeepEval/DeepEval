import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from keras.layers import BatchNormalization, Activation

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32

    # Path 1: Single 1x1 convolution
    branch1 = Conv2D(64, (1, 1), padding='same')(input_shape)
    branch1 = BatchNormalization()(branch1)
    branch1 = Activation('relu')(branch1)

    # Path 2: 1x1 convolution + 1x7 and 7x1 convolutions
    branch2 = Conv2D(64, (1, 1))(input_shape)
    branch2 = BatchNormalization()(branch2)
    branch2 = Activation('relu')(branch2)
    branch2 = Conv2D(64, (1, 7), padding='same')(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = Activation('relu')(branch2)
    branch2 = Conv2D(64, (7, 1), padding='same')(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = Activation('relu')(branch2)

    # Path 3: 1x1 convolution + two sets of 1x7 and 7x1 convolutions
    branch3 = Conv2D(64, (1, 1))(input_shape)
    branch3 = BatchNormalization()(branch3)
    branch3 = Activation('relu')(branch3)
    branch3 = Conv2D(64, (1, 7), padding='same')(branch3)
    branch3 = BatchNormalization()(branch3)
    branch3 = Activation('relu')(branch3)
    branch3 = Conv2D(64, (7, 1), padding='same')(branch3)
    branch3 = BatchNormalization()(branch3)
    branch3 = Activation('relu')(branch3)
    branch3 = Conv2D(64, (7, 1), padding='same')(branch3)
    branch3 = BatchNormalization()(branch3)
    branch3 = Activation('relu')(branch3)

    # Path 4: Average pooling followed by 1x1 convolution
    branch4 = MaxPooling2D((2, 2))(input_shape)
    branch4 = Conv2D(64, (1, 1))(branch4)
    branch4 = BatchNormalization()(branch4)
    branch4 = Activation('relu')(branch4)

    # Concatenate the outputs of all paths
    fused_branch = concatenate([branch1, branch2, branch3, branch4])

    # Flatten the concatenated branch
    flat_layer = Flatten()(fused_branch)

    # Fully connected layer for classification
    output = Dense(10, activation='softmax')(flat_layer)  # Assuming 10 classes

    # Construct the model
    model = Model(inputs=input_shape, outputs=output)

    return model

# Example usage:
model = dl_model()
model.summary()