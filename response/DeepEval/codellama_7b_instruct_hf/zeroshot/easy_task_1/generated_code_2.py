from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    # Create the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Add the first convolutional layer
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)

    # Add the max pooling layer
    pool1 = MaxPooling2D((2, 2))(conv1)

    # Add the second convolutional layer
    conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)

    # Add the flattening layer
    flatten = Flatten()(conv2)

    # Add the fully connected layers
    fc1 = Dense(128, activation='relu')(flatten)
    fc2 = Dense(10, activation='softmax')(fc1)

    # Create the model
    model = Model(inputs=input_layer, outputs=fc2)

    return model