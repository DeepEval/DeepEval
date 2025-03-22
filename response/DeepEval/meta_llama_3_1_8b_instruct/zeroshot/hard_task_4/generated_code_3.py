# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def dl_model():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize the input data
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Define the deep learning model
    inputs = keras.Input(shape=(32, 32, 3))

    # Increase the dimensionality of the input's channels threefold with a 1x1 convolution
    x = layers.Conv2D(3 * 3, (1, 1), padding='same')(inputs)

    # Extract initial features using a 3x3 depthwise separable convolution
    x = layers.SeparableConv2D(3, (3, 3), padding='same')(x)

    # Compute channel attention weights through global average pooling
    channel_weights = layers.GlobalAveragePooling2D()(x)
    channel_weights = layers.Dense(3, activation='relu')(channel_weights)
    channel_weights = layers.Dense(3, activation='sigmoid')(channel_weights)

    # Reshape the channel weights to match the initial features
    channel_weights = layers.Reshape((1, 1, 3))(channel_weights)

    # Multiply the channel weights with the initial features to achieve channel attention weighting
    x = layers.Multiply()([x, channel_weights])

    # Reduce the dimensionality using a 1x1 convolution
    x = layers.Conv2D(3, (1, 1), padding='same')(x)

    # Combine the output with the initial input
    x = layers.Add()([x, inputs])

    # Flattening layer
    x = layers.Flatten()(x)

    # Fully connected layer for classification
    outputs = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Return the constructed model
    return model

# Test the dl_model function
model = dl_model()
model.summary()