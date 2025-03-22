from keras.models import Model
from keras.layers import Input, Conv2D, Dense, Flatten, Add


def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Path 1: Sequential convolutional layers
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)

    # Path 2: Separate convolutional layer
    y = Conv2D(32, (5, 5), activation='relu')(inputs)

    # Add outputs from both paths
    z = Add()([x, y])

    # Fully connected layers
    z = Flatten()(z)
    z = Dense(256, activation='relu')(z)
    outputs = Dense(10, activation='softmax')(z)

    # Model definition
    model = Model(inputs=inputs, outputs=outputs)


    return model