import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Flatten, Reshape
from tensorflow.keras.utils import to_categorical

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Input shape
input_shape = (32, 32, 3)  # Input image shape is 32x32 RGB images

# Model function
def dl_model():
    # Branch 1
    branch1 = tf.keras.Sequential([
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Reshape((128, 1, 1))
    ])

    # Branch 2
    branch2 = tf.keras.Sequential([
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Reshape((128, 1, 1))
    ])

    # Concatenate and process through final dense layer
    x = tf.keras.layers.concatenate([branch1, branch2])
    x = Flatten()(x)
    output = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=[branch1.input, branch2.input], outputs=[output])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Build and return the model
model = dl_model()
model.summary()