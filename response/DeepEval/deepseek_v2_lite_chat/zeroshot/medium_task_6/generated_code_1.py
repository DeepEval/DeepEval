from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, ReLU, Flatten, Dense, Add
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.optimizers import Adam

# Load and prepare the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the model architecture
def dl_model():
    # Create a sequential model
    model = Sequential([
        # Initial convolution
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        BatchNormalization(),
        ReLU(),
        # Convolution block 1
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        ReLU(),
        # Convolution block 2
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        ReLU(),
        # Convolution block 3
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        ReLU(),
        # Add the outputs of the blocks to enhance feature representation
        Add(),
        # Flatten the output for fully connected layers
        Flatten(),
        # Fully connected layers
        Dense(512, activation='relu'),
        Dense(10, activation='softmax'),  # Softmax for multi-class classification
    ])

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Get the model
model = dl_model()