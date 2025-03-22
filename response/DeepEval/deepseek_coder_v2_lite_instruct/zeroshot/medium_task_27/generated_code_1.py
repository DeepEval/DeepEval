import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Add, GlobalAveragePooling2D, Dense, Multiply
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # First convolutional branch (3x3 kernel)
    branch1 = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    branch1 = Conv2D(32, (3, 3), padding='same', activation='relu')(branch1)

    # Second convolutional branch (5x5 kernel)
    branch2 = Conv2D(32, (5, 5), padding='same', activation='relu')(inputs)
    branch2 = Conv2D(32, (5, 5), padding='same', activation='relu')(branch2)

    # Add the outputs of the two branches
    added = Add()([branch1, branch2])

    # Global Average Pooling
    gap = GlobalAveragePooling2D()(added)

    # Fully connected layers
    fc1 = Dense(128, activation='relu')(gap)
    fc2 = Dense(10, activation='softmax')(fc1)

    # Create the model
    model = Model(inputs=inputs, outputs=fc2)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Summary of the model
    model.summary()

    return model

# Create and train the model
model = dl_model()