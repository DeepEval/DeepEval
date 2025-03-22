# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Define the input layer
    input_layer = keras.Input(shape=(32, 32, 3), name="input_layer")

    # Convolutional layer 1
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)

    # Convolutional layer 2
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)

    # Max-pooling layer
    x = layers.MaxPooling2D((2, 2))(x)

    # Add the input layer to the features
    x = layers.Add()([x, input_layer])

    # Flatten the features
    x = layers.Flatten()(x)

    # Fully connected layer 1
    x = layers.Dense(128, activation="relu")(x)

    # Fully connected layer 2
    output_layer = layers.Dense(10, activation="softmax")(x)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Print the model summary
print(model.summary())