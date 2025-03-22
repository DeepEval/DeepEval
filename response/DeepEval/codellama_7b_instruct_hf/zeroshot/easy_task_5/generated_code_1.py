import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    # Create the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Add a 1x1 convolutional layer
    x = Conv2D(32, (1, 1), activation='relu')(input_layer)

    # Add a 3x3 convolutional layer
    x = Conv2D(32, (3, 3), activation='relu')(x)

    # Add a max pooling layer
    x = MaxPooling2D((2, 2))(x)

    # Add a flatten layer
    x = Flatten()(x)

    # Add a fully connected layer
    x = Dense(10, activation='softmax')(x)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=x)

    return model