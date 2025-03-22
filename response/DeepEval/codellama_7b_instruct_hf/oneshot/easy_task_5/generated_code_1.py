import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Reduce the input dimensionality with a 1x1 convolution
    conv1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)

    # Extract features using a 3x3 convolutional layer
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv1)

    # Restore the dimensionality of the feature map with another 1x1 convolution layer
    conv3 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(conv2)

    # Flatten the output
    flatten_layer = Flatten()(conv3)

    # Add a fully connected layer with 10 neurons for classification
    dense = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=dense)

    return model