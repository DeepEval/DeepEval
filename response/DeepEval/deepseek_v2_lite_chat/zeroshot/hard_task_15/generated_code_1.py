import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from tensorflow.keras.optimizers import Adam


def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # One-hot encode the labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)


    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Branch path
    branch_input = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    branch_input = MaxPooling2D(pool_size=(2, 2))(branch_input)
    branch_input = Conv2D(64, (3, 3), activation='relu', padding='same')(branch_input)
    branch_output = Flatten()(branch_input)

    # Main path
    global_avg_pool = GlobalAveragePooling2D()(inputs)
    fully_connected1 = Dense(1024, activation='relu')(global_avg_pool)
    fully_connected1 = Dense(1024, activation='relu')(fully_connected1)

    # Concatenate branch and main outputs
    concat = concatenate([branch_output, fully_connected1])

    # Fully connected layers
    fully_connected2 = Dense(512, activation='relu')(concat)
    output = Dense(10, activation='softmax')(fully_connected2)

    # Model architecture
    model = Model(inputs=inputs, outputs=output)

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Return the model
    return model