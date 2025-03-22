import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense


def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first convolutional layer
    conv1 = Conv2D(32, (3, 3), activation='relu')

    # Define the second convolutional layer
    conv2 = Conv2D(64, (3, 3), activation='relu')

    # Define the third convolutional layer
    conv3 = Conv2D(128, (3, 3), activation='relu')

    # Define the fully connected layers
    fc1 = Dense(128, activation='relu')
    fc2 = Dense(10, activation='softmax')

    # Define the input layer
    input_layer = Input(shape=input_shape)

    # Define the first convolutional layer
    x = conv1(input_layer)

    # Define the second convolutional layer
    x = MaxPooling2D((2, 2))(x)
    x = conv2(x)

    # Define the third convolutional layer
    x = MaxPooling2D((2, 2))(x)
    x = conv3(x)

    # Define the fully connected layers
    x = Flatten()(x)
    x = fc1(x)
    x = fc2(x)

    # Define the output layer
    output_layer = Dense(10, activation='softmax')(x)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

    return model