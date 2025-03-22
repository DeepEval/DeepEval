# Import necessary libraries
from keras.layers import Input, DepthwiseConv2D, Conv2D, MaxPooling2D, Flatten, Dense
from keras.layers import Dropout
from keras.models import Model

# Define the function to create the deep learning model
def dl_model():
    # Define the input shape of the MNIST dataset
    input_shape = (28, 28, 1)

    # Define the input layer
    inputs = Input(shape=input_shape, name='input_layer')

    # Add the depthwise separable convolutional layer
    x = DepthwiseConv2D(kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    x = Dropout(0.2)(x)  # Add dropout layer to mitigate overfitting
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Add the 1x1 convolutional layer for feature extraction
    x = Conv2D(kernel_size=(1, 1), filters=64, activation='relu', padding='same')(x)
    x = Dropout(0.2)(x)  # Add dropout layer to mitigate overfitting
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten the output and add a fully connected layer for classification
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs, name='image_classification_model')

    return model