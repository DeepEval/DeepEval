import keras
from keras import layers
from keras.models import Model
from keras.datasets import cifar10
from keras.utils import to_categorical

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess the data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define the model
def dl_model():
    # Input layer
    inputs = layers.Input(shape=(32, 32, 3))

    # Initial convolutional layer
    x = layers.Conv2D(64, (1, 1), padding="same")(inputs)

    # Branch 1: Local features
    branch1 = layers.Conv2D(128, (3, 3), padding="same")(x)

    # Branch 2: Downsampling and upsampling
    branch2 = layers.MaxPooling2D(2, 2)(x)
    branch2 = layers.Conv2D(128, (3, 3), padding="same")(branch2)
    branch2 = layers.UpSampling2D(2)(branch2)

    # Branch 3: Downsampling and upsampling
    branch3 = layers.MaxPooling2D(2, 2)(x)
    branch3 = layers.Conv2D(128, (3, 3), padding="same")(branch3)
    branch3 = layers.UpSampling2D(2)(branch3)

    # Concatenate branches
    merged = layers.concatenate([branch1, branch2, branch3])

    # Final convolutional layer
    merged = layers.Conv2D(64, (1, 1), padding="same")(merged)

    # Fully connected layers
    merged = layers.Flatten()(merged)
    merged = layers.Dense(32, activation="relu")(merged)
    merged = layers.Dense(10, activation="softmax")(merged)

    # Model definition
    model = Model(inputs, merged)

    # Compilation
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Training
    model.fit(x_train, y_train, epochs=10, batch_size=64)

    # Evaluation
    model.evaluate(x_test, y_test, verbose=2)

    return model

# Construct and return the model
model = dl_model()