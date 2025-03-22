from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    ReLU,
    SeparableConv2D,
    MaxPooling2D,
    Conv2D,
    Add,
    Flatten,
    Dense,
)
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Preprocess data
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype("float32") / 255.0
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype("float32") / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Input layer
    img_input = Input(shape=(28, 28, 1))

    # Main path
    x = img_input
    for _ in range(2):
        x = ReLU()(x)
        x = SeparableConv2D(filters=32, kernel_size=3, padding="same")(x)
        x = MaxPooling2D(pool_size=2, strides=2)(x)

    # Branch path
    y = Conv2D(filters=32, kernel_size=1, padding="same")(img_input)

    # Combine main and branch paths
    combined = Add()([x, y])

    # Output layer
    flattened = Flatten()(combined)
    output = Dense(10, activation="softmax")(flattened)

    # Create model
    model = Model(img_input, output)

    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train and evaluate the model
model.fit(x_train, y_train, epochs=10)
loss, accuracy = model.evaluate(x_test, y_test)
print("Test accuracy:", accuracy)