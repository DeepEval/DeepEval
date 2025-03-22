# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to the range [0, 1]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Convert class labels to categorical labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

def dl_model():
    # Define the initial convolutional layer
    inputs = keras.Input(shape=(32, 32, 3))
    initial_conv = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    initial_conv = layers.BatchNormalization()(initial_conv)

    # Define the first parallel block
    conv_block1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(initial_conv)
    conv_block1 = layers.BatchNormalization()(conv_block1)
    conv_block1 = layers.ReLU()(conv_block1)

    # Define the second parallel block
    conv_block2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv_block1)
    conv_block2 = layers.BatchNormalization()(conv_block2)
    conv_block2 = layers.ReLU()(conv_block2)

    # Define the third parallel block
    conv_block3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv_block2)
    conv_block3 = layers.BatchNormalization()(conv_block3)
    conv_block3 = layers.ReLU()(conv_block3)

    # Add the outputs of the parallel blocks
    added_output = layers.Add()([initial_conv, conv_block1, conv_block2, conv_block3])

    # Flatten the output
    flattened = layers.Flatten()(added_output)

    # Define the fully connected layers
    fc1 = layers.Dense(128, activation='relu')(flattened)
    fc2 = layers.Dense(64, activation='relu')(fc1)
    outputs = layers.Dense(10, activation='softmax')(fc2)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Create the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))