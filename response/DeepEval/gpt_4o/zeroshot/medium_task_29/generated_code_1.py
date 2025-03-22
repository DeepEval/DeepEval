import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer for CIFAR-10 images
    input_layer = Input(shape=(32, 32, 3))

    # First convolutional layer
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)

    # MaxPooling layers with varying window sizes
    pool_1x1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(x)
    pool_2x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    pool_4x4 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(x)

    # Flatten each pooled output
    flat_1x1 = Flatten()(pool_1x1)
    flat_2x2 = Flatten()(pool_2x2)
    flat_4x4 = Flatten()(pool_4x4)

    # Concatenate flattened layers
    concatenated = Concatenate()([flat_1x1, flat_2x2, flat_4x4])

    # Fully connected layers
    fc1 = Dense(128, activation='relu')(concatenated)
    fc2 = Dense(64, activation='relu')(fc1)

    # Output layer for 10 classes
    output_layer = Dense(10, activation='softmax')(fc2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of how to load data and compile the model
def compile_and_train_model():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    model = dl_model()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# Uncomment the line below to compile and train the model
# compile_and_train_model()