from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 64)

    # Define the first convolutional layer
    conv1 = Conv2D(32, (3, 3), activation='relu')

    # Define the first pooling layer
    pool1 = MaxPooling2D((2, 2))

    # Define the second convolutional layer
    conv2 = Conv2D(64, (3, 3), activation='relu')

    # Define the second pooling layer
    pool2 = MaxPooling2D((2, 2))

    # Define the third convolutional layer
    conv3 = Conv2D(64, (3, 3), activation='relu')

    # Define the third pooling layer
    pool3 = MaxPooling2D((2, 2))

    # Define the flattening layer
    flatten = Flatten()

    # Define the first fully connected layer
    fc1 = Dense(128, activation='relu')

    # Define the second fully connected layer
    fc2 = Dense(10, activation='softmax')

    # Define the input layer
    input_layer = Input(shape=input_shape)

    # Define the output layer
    output_layer = fc2(fc1(flatten(pool3(conv3(pool2(conv2(pool1(conv1(input_layer)))))))))

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model