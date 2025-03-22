from keras.applications import VGG16
from keras.layers import Input, Concatenate, Flatten, Dense, Softmax
from keras.models import Model


def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first path with a 1x1 convolution
    x = Input(shape=input_shape)
    y = Conv2D(32, (1, 1), activation='relu')(x)

    # Define the second path with two 3x3 convolutions stacked after a 1x1 convolution
    y = Conv2D(32, (1, 1), activation='relu')(y)
    y = Conv2D(64, (3, 3), activation='relu')(y)
    y = Conv2D(64, (3, 3), activation='relu')(y)

    # Define the third path with a single 3x3 convolution followed by a 1x1 convolution
    y = Conv2D(32, (1, 1), activation='relu')(y)
    y = Conv2D(64, (3, 3), activation='relu')(y)

    # Define the fourth path with max pooling followed by a 1x1 convolution
    y = MaxPooling2D((2, 2))(y)
    y = Conv2D(64, (1, 1), activation='relu')(y)

    # Concatenate the outputs from the four paths
    y = Concatenate()([x, y])

    # Flatten the output and pass through a dense layer with 128 units
    y = Flatten()(y)
    y = Dense(128, activation='relu')(y)

    # Output the final layer with softmax activation
    y = Dense(10, activation='softmax')(y)

    # Create the model
    model = Model(inputs=x, outputs=y)

    # Compile the model with a loss function and optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model