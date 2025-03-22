from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

def dl_model():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape the input data to add a channel dimension
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

    # Preprocess the input data
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Create the model
    model = keras.Sequential(
        [
            layers.InputLayer(input_shape=(28, 28, 1)),
            layers.Conv2D(
                filters=32,
                kernel_size=3,
                strides=1,
                padding="same",
                use_bias=False,
                depthwise_initializer="he_normal",
            ),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.2),
            layers.Conv2D(
                filters=32,
                kernel_size=1,
                strides=1,
                padding="same",
                use_bias=True,
                kernel_initializer="he_normal",
            ),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(10, kernel_initializer="glorot_normal"),
        ]
    )

    return model