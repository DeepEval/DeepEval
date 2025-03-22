import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Concatenate, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, AveragePooling2D

# Constants
IMG_ROWS, IMG_COLS, IMG_CHANNELS = 32, 32, 3
CLASSES = 10
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, IMG_CHANNELS)

def dl_model():
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize the images to [-1, 1]
    x_train, x_test = x_train / 127.5 - 1.0, x_test / 127.5 - 1.0

    # Define the input shape
    input_layer = Input(shape=INPUT_SHAPE)

    # Split input into three groups
    x = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)

    # Define the first depthwise separable convolution
    def depthwise_conv2d(filters, size, strides=1, padding='same', name=None):
        return Conv2D(filters, size, strides=strides, padding=padding, data_format='channels_last', name=name), \
               BatchNormalization(momentum=0.9, epsilon=0.001)

    # Apply depthwise separable convolutions
    x[0] = depthwise_conv2d(64, (1, 1))(x[0])
    x[1] = depthwise_conv2d(64, (3, 3))(x[1])
    x[2] = depthwise_conv2d(64, (5, 5))(x[2])

    # Batch normalization and activation
    x = tf.keras.layers.LeakyReLU()(x[0])
    x = tf.keras.layers.LeakyReLU()(x[1])
    x = tf.keras.layers.LeakyReLU()(x[2])

    # Concatenate the outputs
    x = Concatenate(axis=-1)(x)

    # Second block for feature extraction
    x = AveragePooling2D(pool_size=(3, 3))(x)
    x = Conv2D(64, (1, 7), padding='same')(x)
    x = Conv2D(64, (7, 1), padding='same')(x)
    x = depthwise_conv2d(64, (3, 3))(x)
    x = depthwise_conv2d(64, (1, 7), strides=2, padding='same')(x)
    x = depthwise_conv2d(64, (7, 1), strides=2, padding='same')(x)
    x = depthwise_conv2d(64, (3, 3))(x)

    # Classification block
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    output = Dense(CLASSES, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Build and train the model
model = dl_model()
model.fit(x_train, y_train, batch_size=64, epochs=30, validation_data=(x_test, y_test))