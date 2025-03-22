import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Add, Concatenate
from tensorflow.keras.models import Model


def dl_model():
    # Load the CIFAR-10 dataset
    (cifar_train_images, cifar_train_labels), (cifar_test_images, cifar_test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    cifar_train_images, cifar_test_images = cifar_train_images / 255.0, cifar_test_images / 255.0

    # Define the input shape
    input_shape = (32, 32, 3)

    # Create the main path of the model
    input_main = Input(shape=input_shape)
    x = Conv2D(64, (1, 1), padding='same')(input_main)  # 1x1 convolution

    x = Conv2D(64, (1, 7), padding='same')(x)  # 1x7 convolution
    x = Conv2D(64, (7, 1), padding='same')(x)  # 7x1 convolution
    x = Conv2D(64, (1, 1), padding='same')(x)  # Another 1x1 convolution

    x = Add()([input_main, x])  # Add the branch directly connected to the input

    # Flatten the output to feed it into the fully connected layers
    x = layers.Flatten()(x)

    # Output for the main path
    x = Conv2D(64, (1, 1), padding='same')(x)  # Align the output dimensions with the input image's channel
    x = layers.Flatten()(x)

    # Fully connected layers for classification
    x = layers.Dense(128, activation='relu')(x)
    predictions = layers.Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10

    # Create the model
    model = Model(inputs=input_main, outputs=predictions)

    # Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model