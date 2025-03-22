import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, concatenate, Dense, Reshape
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical


def dl_model():
    # Parameters
    input_shape = (28, 28, 1)  # MNIST images are 28x28 pixels
    num_classes = 10  # 10 classes for MNIST (0-9)

    # Block 1: Convolution and Pooling Layers
    block1_input = Input(shape=input_shape)
    conv1_1 = Conv2D(32, (1, 1), activation='relu', padding='same')(block1_input)
    conv1_2 = Conv2D(32, (2, 2), activation='relu', padding='same')(conv1_1)
    conv1_3 = Conv2D(32, (4, 4), activation='relu', padding='same')(conv1_1)
    avg1 = MaxPooling2D((2, 2), padding='same')(conv1_3)
    flat1 = Flatten()(avg1)

    # Concatenate the outputs of all pooling paths
    concat1 = concatenate([flat1, flat1, flat1])

    # Block 2: Feature Extraction Branches
    branch1 = Conv2D(64, (1, 1), activation='relu', padding='same')(block1_input)
    branch2 = Conv2D(64, (3, 3), activation='relu', padding='same')(block1_input)
    branch3 = Conv2D(64, (1, 7), activation='relu', padding='same', name='branch3')(block1_input)
    branch4 = Conv2D(64, (7, 1), activation='relu', padding='same', name='branch4')(block1_input)
    avg2 = MaxPooling2D((2, 2), padding='same')(concat1)

    # Process each branch
    branch1_out = Flatten()(branch1)
    branch2_out = Flatten()(branch2)
    branch3_out = Flatten()(branch3)
    branch4_out = Flatten()(branch4)

    # Concatenate the outputs of all branches
    concat2 = concatenate([branch1_out, branch2_out, branch3_out, branch4_out])

    # Fully connected layers
    fc1 = Dense(512, activation='relu')(concat2)
    output = Dense(num_classes, activation='softmax')(fc1)

    # Model
    model = Model(inputs=block1_input, outputs=output)

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# Build the model
model = dl_model()