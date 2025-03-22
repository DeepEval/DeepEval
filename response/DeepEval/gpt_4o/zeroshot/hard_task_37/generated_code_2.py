from tensorflow.keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

def conv_block(input_tensor):
    # Block with three convolutional layers, each creating a separate path
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)

    # Parallel branch connecting the input through a convolutional layer
    shortcut = Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)

    # Adding all paths together
    added = Add()([conv1, conv2, conv3, shortcut])
    return added

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # First block
    block1 = conv_block(input_layer)

    # Second block
    block2 = conv_block(input_layer)

    # Concatenating the outputs of both blocks
    concatenated = Concatenate()([block1, block2])

    # Flattening layer
    flat = Flatten()(concatenated)

    # Fully connected layer
    dense = Dense(128, activation='relu')(flat)

    # Output layer for classification (10 classes for MNIST)
    output_layer = Dense(10, activation='softmax')(dense)

    # Creating the model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Example of how to compile and fit the model
if __name__ == "__main__":
    # Load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Build and compile the model
    model = dl_model()
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))