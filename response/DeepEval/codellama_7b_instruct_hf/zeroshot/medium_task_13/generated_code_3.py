from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    # Create the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Convolutional layer 1
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    pool1 = MaxPooling2D((2, 2))(conv1)

    # Convolutional layer 2
    conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)

    # Convolutional layer 3
    conv3 = Conv2D(128, (3, 3), activation='relu')(pool2)
    pool3 = MaxPooling2D((2, 2))(conv3)

    # Flatten the output of the last convolutional layer
    flat = Flatten()(pool3)

    # Dense layer 1
    dense1 = Dense(128, activation='relu')(flat)

    # Dense layer 2
    output = Dense(10, activation='softmax')(dense1)

    # Create the model
    model = Model(inputs=input_layer, outputs=output)

    return model