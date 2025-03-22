import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer for 28x28 grayscale images
    inputs = Input(shape=(28, 28, 1))

    # 1x1 Convolutional layer to increase dimensionality
    x = Conv2D(filters=32, kernel_size=(1, 1), strides=(2, 2), activation='relu')(inputs)

    # 3x3 Depthwise separable convolutional layer for feature extraction
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)

    # 1x1 Convolutional layer to reduce dimensionality
    x = Conv2D(filters=10, kernel_size=(1, 1), strides=(2, 2), activation='relu')(x)

    # Flatten the output
    x = Flatten()(x)

    # Fully connected layer for classification
    outputs = Dense(units=10, activation='softmax')(x)

    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Example of loading the MNIST dataset
def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255.0
    x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.0
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    return (x_train, y_train), (x_test, y_test)

# Example usage
if __name__ == "__main__":
    model = dl_model()
    print(model.summary())
    
    # Load the data
    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    # Fit the model on the training data
    model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)