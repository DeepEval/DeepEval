# Import necessary packages
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, MaxPooling2D, Flatten, Dense
from keras.models import Model
from keras.layers import concatenate
from keras.regularizers import l2
from keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
import numpy as np
from tensorflow.keras.utils import to_categorical

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Convert class labels to categorical labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define the input shape
input_shape = (32, 32, 3)

# Define the dl_model function
def dl_model():
    # Define the input layer
    inputs = Input(shape=input_shape)

    # Define the first sequential block
    block1 = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(0.01))(inputs)
    block1 = BatchNormalization()(block1)
    block1 = ReLU()(block1)
    block1 = MaxPooling2D(pool_size=(2, 2))(block1)

    # Define the second sequential block
    block2 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.01))(block1)
    block2 = BatchNormalization()(block2)
    block2 = ReLU()(block2)
    block2 = MaxPooling2D(pool_size=(2, 2))(block2)

    # Define the third sequential block
    block3 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.01))(block2)
    block3 = BatchNormalization()(block3)
    block3 = ReLU()(block3)
    block3 = MaxPooling2D(pool_size=(2, 2))(block3)

    # Define the parallel branch
    parallel_branch = Conv2D(128, (3, 3), padding='same')(block3)
    parallel_branch = BatchNormalization()(parallel_branch)
    parallel_branch = ReLU()(parallel_branch)

    # Add the outputs of the three blocks
    concatenated_output = concatenate([block1, block2, block3, parallel_branch], axis=-1)

    # Flatten the output
    flattened_output = Flatten()(concatenated_output)

    # Define the first fully connected layer
    fc1 = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(flattened_output)

    # Define the second fully connected layer
    fc2 = Dense(10, activation='softmax', kernel_regularizer=l2(0.01))(fc1)

    # Define the model
    model = Model(inputs=inputs, outputs=fc2)

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

    return model

# Test the dl_model function
model = dl_model()
model.summary()