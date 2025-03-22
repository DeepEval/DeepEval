import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, LayerNormalization, Add, Dense, Flatten


def dl_model():
    
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize the pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # One-hot encode the labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Define the input shape
    input_shape = (32, 32, 3)  # Each image is 32x32 pixels

    # Input layer
    input_layer = Input(shape=input_shape)

    # Spatial feature extraction with depthwise separable convolution
    x = Conv2D(32, (7, 7), strides=(2, 2), padding="same", activation="relu")(input_layer)
    x = LayerNormalization()(x)

    # Max pooling layer for downsampling
    x = tf.keras.layers.MaxPooling2D(pool_size=(5, 5))(x)

    # Pass through two fully connected layers
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)

    # Channel-wise feature transformation
    x = Dense(256)(x)
    x = Dense(128, activation="relu")(x)

    # Add the original input with the processed features
    output = Add()([input_layer, x])

    # Final classification with two fully connected layers
    output = Dense(1024, activation="relu")(output)
    output = Dense(512, activation="relu")(output)
    output = Dense(10, activation="softmax")(output)

    # Model
    model = Model(inputs=input_layer, outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model