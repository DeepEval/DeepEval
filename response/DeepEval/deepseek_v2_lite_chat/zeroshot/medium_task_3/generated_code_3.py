import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

def dl_model():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the images to the 0-1 range
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Reshape the data
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))

    # Input layers
    input_A = Input(shape=(28, 28, 1))
    input_B = Input(shape=(28, 28, 1))

    # Path 1: Convolutional layers
    conv_1 = Conv2D(32, (3, 3), activation='relu')(input_A)
    maxpool_1 = MaxPooling2D()(conv_1)

    # Path 2: Convolutional layers
    conv_2 = Conv2D(64, (3, 3), activation='relu')(input_B)
    maxpool_2 = MaxPooling2D()(conv_2)

    # Flatten the outputs
    flat_1 = Flatten()(maxpool_1)
    flat_2 = Flatten()(maxpool_2)

    # Combine the outputs
    combined = Add()([flat_1, flat_2])

    # Fully connected layer
    dense = Dense(10, activation='softmax')(combined)

    # Model
    model = Model(inputs=[input_A, input_B], outputs=[dense])

    # Compile the model
    model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    return model

# Create and return the model
model = dl_model()
model.summary()