from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define the model architecture
def dl_model():
    # Input layer
    inputs = layers.Input(shape=(32, 32, 3))

    # Block 1
    block1 = layers.Conv2D(32, (3, 3), padding="same")(inputs)
    block1 = layers.BatchNormalization()(block1)
    block1 = layers.Activation("relu")(block1)
    block1 = layers.Conv2D(32, (3, 3), padding="same")(block1)
    block1 = layers.BatchNormalization()(block1)
    block1 = layers.Activation("relu")(block1)
    block1 = layers.MaxPooling2D(pool_size=(2, 2))(block1)

    # Block 2
    block2 = layers.Conv2D(64, (3, 3), padding="same")(block1)
    block2 = layers.BatchNormalization()(block2)
    block2 = layers.Activation("relu")(block2)
    block2 = layers.Conv2D(64, (3, 3), padding="same")(block2)
    block2 = layers.BatchNormalization()(block2)
    block2 = layers.Activation("relu")(block2)
    block2 = layers.MaxPooling2D(pool_size=(2, 2))(block2)

    # Block 3
    block3 = layers.Conv2D(128, (3, 3), padding="same")(block2)
    block3 = layers.BatchNormalization()(block3)
    block3 = layers.Activation("relu")(block3)
    block3 = layers.Conv2D(128, (3, 3), padding="same")(block3)
    block3 = layers.BatchNormalization()(block3)
    block3 = layers.Activation("relu")(block3)
    block3 = layers.MaxPooling2D(pool_size=(2, 2))(block3)

    # Flatten and fully connected layers
    flatten = layers.Flatten()(block3)
    outputs = layers.Dense(10, activation="softmax")(flatten)

    # Model definition
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Compile and train the model
model = dl_model()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print("Test accuracy:", accuracy)