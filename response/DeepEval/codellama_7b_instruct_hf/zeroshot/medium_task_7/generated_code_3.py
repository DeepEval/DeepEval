from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First convolutional layer
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    maxpool1 = MaxPooling2D((2, 2))(conv1)

    # Second convolutional layer
    conv2 = Conv2D(64, (3, 3), activation='relu')(maxpool1)
    maxpool2 = MaxPooling2D((2, 2))(conv2)

    # Third convolutional layer
    conv3 = Conv2D(128, (3, 3), activation='relu')(maxpool2)
    maxpool3 = MaxPooling2D((2, 2))(conv3)

    # Adding the outputs of the first two convolutional layers
    added_layers = Add()([conv1, conv2])

    # Adding the output of the third convolutional layer
    added_layers = Add()([added_layers, conv3])

    # Separate convolutional layer
    sep_conv = Conv2D(256, (3, 3), activation='relu')(input_layer)

    # Adding the outputs of all paths
    added_layers = Add()([added_layers, sep_conv])

    # Flattening the output
    flattened = Flatten()(added_layers)

    # Dense layers
    dense1 = Dense(128, activation='relu')(flattened)
    dense2 = Dense(10, activation='softmax')(dense1)

    # Creating the model
    model = Model(inputs=input_layer, outputs=dense2)

    # Compiling the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model