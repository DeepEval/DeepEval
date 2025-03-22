import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, concatenate, Dense, Reshape
from tensorflow.keras.models import Model

def dl_model():
    # Block 1: Extract multi-scale features
    input_layer = Input(shape=(28, 28, 1))  # Input shape should be (28, 28, 1) for MNIST

    # Path 1: Max pooling with different scales
    path1 = Conv2D(64, (1, 1), padding='same')(input_layer)
    path1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(path1)
    path1 = Dropout(0.25)(path1)
    path1 = Flatten()(path1)

    # Path 2: Max pooling with intermediate convolutions
    path2 = Conv2D(64, (1, 1))(input_layer)
    path2 = Conv2D(64, (1, 7), padding='same')(path2)
    path2 = Conv2D(64, (7, 1), padding='same')(path2)
    path2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(path2)
    path2 = Dropout(0.25)(path2)
    path2 = Flatten()(path2)

    # Path 3: Complex convolutions for deeper features
    path3 = Conv2D(64, (1, 1))(input_layer)
    path3 = Conv2D(64, (7, 1))(path3)
    path3 = Conv2D(64, (1, 7))(path3)
    path3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(path3)
    path3 = Dropout(0.25)(path3)
    path3 = Flatten()(path3)

    # Path 4: Compressed feature extraction with pooling and convolution
    path4 = Conv2D(64, (1, 1))(input_layer)
    path4 = AveragePooling2D(pool_size=(2, 2))(path4)
    path4 = Conv2D(64, (1, 1))(path4)
    path4 = Flatten()(path4)

    # Concatenate the outputs from all paths
    concatenated = concatenate([path1, path2, path3, path4])

    # Block 2: Final classification using fully connected layers
    dense1 = Dense(512, activation='relu')(concatenated)
    dense1 = Dropout(0.5)(dense1)

    reshape = Reshape((4, 4))(dense1)
    dense2 = Dense(256, activation='relu')(reshape)
    dense2 = Dropout(0.5)(dense2)

    output = Dense(10, activation='softmax')(dense2)  # Output layer for 10 classes (0-9 digits)

    model = Model(inputs=input_layer, outputs=output)

    return model

# Check the model summary
model = dl_model()
model.summary()