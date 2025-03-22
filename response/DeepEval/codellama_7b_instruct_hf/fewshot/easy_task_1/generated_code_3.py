import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Define the first convolutional layer
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)

    # Define the max pooling layer
    max_pool1 = MaxPooling2D((2, 2))(conv1)

    # Define the second convolutional layer
    conv2 = Conv2D(64, (3, 3), activation='relu')(max_pool1)

    # Flatten the feature maps into a one-dimensional vector
    flattened = Flatten()(conv2)

    # Define the first fully connected layer
    dense1 = Dense(64, activation='relu')(flattened)

    # Define the second fully connected layer
    dense2 = Dense(10, activation='softmax')(dense1)

    # Define the output layer
    output_layer = Dense(10, activation='softmax')(dense2)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model