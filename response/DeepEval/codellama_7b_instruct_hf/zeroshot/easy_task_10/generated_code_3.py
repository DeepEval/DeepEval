from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model
from keras.applications.mnist import MNIST


def dl_model():

    # Load the MNIST dataset
    (X_train, y_train), (X_test, y_test) = MNIST.load_data()

    # Normalize the inputs
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the first convolutional layer
    conv1 = Conv2D(32, (3, 3), activation='relu')

    # Define the second convolutional layer
    conv2 = Conv2D(64, (3, 3), activation='relu')

    # Define the depthwise separable convolutional layer
    depthwise_conv = Conv2D(64, (3, 3), activation='relu', depthwise=True)

    # Define the third convolutional layer
    conv3 = Conv2D(64, (3, 3), activation='relu')

    # Define the flattening layer
    flatten = Flatten()

    # Define the fully connected layer
    fc = Dense(10, activation='softmax')

    # Define the model
    model = Model(inputs=Input(shape=input_shape), outputs=fc)

    # Add the layers to the model
    model.add(conv1)
    model.add(MaxPooling2D((2, 2)))
    model.add(conv2)
    model.add(MaxPooling2D((2, 2)))
    model.add(depthwise_conv)
    model.add(MaxPooling2D((2, 2)))
    model.add(conv3)
    model.add(flatten)
    model.add(fc)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test))

    return model