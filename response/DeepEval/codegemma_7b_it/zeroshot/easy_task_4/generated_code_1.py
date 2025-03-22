from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the data to add an extra dimension for channels
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Create the model using Functional APIs
def dl_model():

    # Input layer
    inputs = layers.Input(shape=(28, 28, 1))

    # First block: convolutional layers followed by max pooling
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Second block: convolutional layers followed by max pooling
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten the feature maps
    x = layers.Flatten()(x)

    # Fully connected layers
    x = layers.Dense(units=128, activation="relu")(x)
    x = layers.Dense(units=64, activation="relu")(x)
    outputs = layers.Dense(units=10, activation="softmax")(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# Compile and train the model
model = dl_model()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print("Test accuracy:", accuracy)