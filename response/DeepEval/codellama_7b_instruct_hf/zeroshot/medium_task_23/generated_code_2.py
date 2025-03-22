from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.applications.inception_v3 import InceptionV3

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Path 1: single 1x1 convolution
    path1 = Conv2D(32, (1, 1), activation='relu')(input_layer)

    # Path 2: 1x1 convolution followed by 1x7 and 7x1 convolutions
    path2 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    path2 = Conv2D(32, (1, 7), activation='relu')(path2)
    path2 = Conv2D(32, (7, 1), activation='relu')(path2)

    # Path 3: 1x1 convolution followed by two sets of 1x7 and 7x1 convolutions
    path3 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    path3 = Conv2D(32, (1, 7), activation='relu')(path3)
    path3 = Conv2D(32, (7, 1), activation='relu')(path3)
    path3 = Conv2D(32, (1, 7), activation='relu')(path3)
    path3 = Conv2D(32, (7, 1), activation='relu')(path3)

    # Path 4: average pooling followed by 1x1 convolution
    path4 = MaxPooling2D((2, 2))(input_layer)
    path4 = Conv2D(32, (1, 1), activation='relu')(path4)

    # Fuse the outputs of the paths through concatenation
    x = Concatenate()([path1, path2, path3, path4])

    # Flatten the output
    x = Flatten()(x)

    # Pass the output through a fully connected layer for classification
    x = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=x)

    # Compile the model with a loss function and an optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model