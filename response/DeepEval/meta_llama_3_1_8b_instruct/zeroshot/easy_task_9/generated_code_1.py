# Import necessary packages
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, Conv2DTranspose, Add, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

def dl_model():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape input data to 4D
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))

    # Normalize pixel values to be between 0 and 1
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Define input layer
    input_layer = Input(shape=(28, 28, 1))

    # 1x1 convolutional layer to increase dimensionality
    conv1 = Conv2D(32, (1, 1), activation='relu')(input_layer)

    # 3x3 depthwise separable convolutional layer for feature extraction
    dw_conv1 = DepthwiseConv2D((3, 3), activation='relu', padding='same')(conv1)
    dw_conv1 = Conv2D(32, (1, 1), activation='relu')(dw_conv1)

    # 1x1 convolutional layer to reduce dimensionality
    conv2 = Conv2D(16, (1, 1), activation='relu')(dw_conv1)

    # Add the output from the 1x1 convolutional layer to the original input layer
    merged = Add()([conv2, input_layer])

    # Flatten the output
    flatten = Flatten()(merged)

    # Fully connected layer to generate the final classification probabilities
    output = Dense(10, activation='softmax')(flatten)

    # Define the model
    model = Model(inputs=input_layer, outputs=output)

    # Compile the model
    model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Test the dl_model function
model = dl_model()
print(model.summary())