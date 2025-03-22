import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Reshape data
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Define the model
def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Branch 1
    branch1 = Conv2D(32, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=l2(0.001))(input_layer)
    branch1 = DepthwiseConv2D((3, 3), activation='relu', depth_multiplier=1)(branch1)
    branch1 = Conv2D(64, (1, 1), activation='relu', padding='same')(branch1)
    branch1 = Dropout(0.5)(branch1)

    # Branch 2
    branch2 = Conv2D(32, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=l2(0.001))(input_layer)
    branch2 = DepthwiseConv2D((3, 3), activation='relu', depth_multiplier=1)(branch2)
    branch2 = Conv2D(64, (1, 1), activation='relu', padding='same')(branch2)
    branch2 = Dropout(0.5)(branch2)

    # Branch 3
    branch3 = Conv2D(32, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=l2(0.001))(input_layer)
    branch3 = DepthwiseConv2D((3, 3), activation='relu', depth_multiplier=1)(branch3)
    branch3 = Conv2D(64, (1, 1), activation='relu', padding='same')(branch3)
    branch3 = Dropout(0.5)(branch3)

    # Concatenate branches
    concatenated = concatenate([branch1, branch2, branch3])

    # Fully connected layers
    fc1 = Flatten()(concatenated)
    fc1 = Dense(512, activation='relu')(fc1)
    dropout1 = Dropout(0.5)(fc1)

    output = Dense(10, activation='softmax')(dropout1)

    # Model
    model = Model(inputs=input_layer, outputs=output)

    # Compile the model
    model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

    return model

# Build the model
model = dl_model()
model.summary()