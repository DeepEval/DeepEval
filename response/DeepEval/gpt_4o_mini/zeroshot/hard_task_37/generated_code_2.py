import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(28, 28, 1))

    # First parallel branch
    def create_branch(input_tensor):
        x = layers.Conv2D(32, (3, 3), activation='relu')(input_tensor)
        x = layers.Conv2D(32, (3, 3), activation='relu')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu')(x)
        return x

    # Create the two branches
    branch1 = create_branch(input_layer)
    branch2 = create_branch(input_layer)

    # Adding a parallel connection with a convolutional layer
    parallel_connection = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)

    # Combine the outputs of the branches and the parallel connection
    combined = layers.add([branch1, branch2, parallel_connection])

    # Flatten the combined outputs
    flatten = layers.Flatten()(combined)

    # Fully connected layer
    dense = layers.Dense(128, activation='relu')(flatten)

    # Output layer for classification (10 classes for MNIST)
    output_layer = layers.Dense(10, activation='softmax')(dense)

    # Construct the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255.0
x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.0
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Create the model
model = dl_model()

# Summary of the model
model.summary()

# Optionally, train the model (commented out)
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))