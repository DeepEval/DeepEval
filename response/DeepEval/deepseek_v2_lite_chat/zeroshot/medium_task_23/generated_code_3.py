from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, concatenate, AveragePooling2D, Flatten, Dense

def dl_model():
    # Input shape
    input_shape = (24, 24, 3)  # CIFAR-10 images are 32x32, but we're using a smaller subset

    # Branch 1: Single 1x1 convolution
    branch1 = Conv2D(32, (1, 1), activation='relu')(input_shape)

    # Branch 2: 1x1 convolution followed by 1x7 and 7x1 convolutions
    branch2 = Conv2D(32, (1, 1), activation='relu')(branch1)
    branch2 = Conv2D(32, (1, 7), padding='valid')(branch2)
    branch2 = Conv2D(64, (7, 1), padding='valid')(branch2)

    # Branch 3: 1x1 convolution followed by two sets of 1x7 and 7x1 convolutions
    branch3 = Conv2D(64, (1, 1), activation='relu')(input_shape)
    branch3 = Conv2D(64, (1, 7), padding='valid')(branch3)
    branch3 = Conv2D(64, (7, 1), padding='valid')(branch3)

    # Branch 4: Average pooling
    branch4 = AveragePooling2D(pool_size=(3, 3))(input_shape)

    # Concatenate the outputs from all branches
    concat = concatenate([branch2, branch3, branch4])

    # Flatten and pass through a fully connected layer
    flat = Flatten()(concat)
    output = Dense(10, activation='softmax')(flat)  # Assuming 10 classes for CIFAR-10

    # Create the model
    model = Model(inputs=input_shape, outputs=output)

    return model

# Instantiate the model
model = dl_model()

# Display the model summary
model.summary()