from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first convolutional layer
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_shape)
    # Define the second convolutional layer
    conv2 = Conv2D(32, (3, 3), activation='relu')(conv1)
    # Define the third convolutional layer
    conv3 = Conv2D(32, (3, 3), activation='relu')(conv2)

    # Define the first fully connected layer
    fc1 = Flatten()(conv3)
    fc1 = Dense(128, activation='relu')(fc1)

    # Define the second fully connected layer
    fc2 = Dense(10, activation='softmax')(fc1)

    # Define the model
    model = Model(inputs=input_shape, outputs=fc2)

    return model