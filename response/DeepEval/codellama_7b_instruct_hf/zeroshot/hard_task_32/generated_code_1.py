import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, DepthwiseConv2D, SeparableConv2D, Dropout, Flatten, Dense
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the three branches of the model
    branch1 = SeparableConv2D(32, (3, 3), activation='relu')(input_shape)
    branch1 = Dropout(0.2)(branch1)
    branch1 = Conv2D(32, (3, 3), activation='relu')(branch1)
    branch1 = Dropout(0.2)(branch1)

    branch2 = SeparableConv2D(64, (3, 3), activation='relu')(input_shape)
    branch2 = Dropout(0.2)(branch2)
    branch2 = Conv2D(64, (3, 3), activation='relu')(branch2)
    branch2 = Dropout(0.2)(branch2)

    branch3 = SeparableConv2D(128, (3, 3), activation='relu')(input_shape)
    branch3 = Dropout(0.2)(branch3)
    branch3 = Conv2D(128, (3, 3), activation='relu')(branch3)
    branch3 = Dropout(0.2)(branch3)

    # Concatenate the outputs of the three branches
    x = np.concatenate([branch1, branch2, branch3], axis=3)

    # Add a flatten layer to transform the output to a 1d vector
    x = Flatten()(x)

    # Add two fully connected layers to generate the final output
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=input_shape, outputs=x)

    # Compile the model with a loss function and an optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model