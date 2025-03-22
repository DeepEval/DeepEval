import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():

    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Reduce dimensionality with a 1x1 convolution
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation="relu")(input_layer)

    # Extract features using a 3x3 convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation="relu")(conv1)

    # Restore dimensionality with another 1x1 convolution layer
    conv3 = Conv2D(filters=128, kernel_size=(1, 1), activation="relu")(conv2)

    # Flatten the output
    flatten = Flatten()(conv3)

    # Fully connected layer with 10 neurons for classification
    output_layer = Dense(units=10, activation="softmax")(flatten)

    # Create the model
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    return model