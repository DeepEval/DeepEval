from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.applications.mnist import MNIST


def dl_model():
    
    # Load the MNIST dataset
    (X_train, y_train), (X_test, y_test) = MNIST(shape=(28, 28, 1), classes=10).load_data()

    # Normalize the input data
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Define the first convolutional layer
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)

    # Define the second convolutional layer
    conv2 = Conv2D(32, (3, 3), activation='relu')(conv1)

    # Define the first path
    first_path = conv1 + MaxPooling2D((2, 2))(conv2)

    # Define the second path
    second_path = input_layer

    # Define the addition layer
    addition_layer = Add()([first_path, second_path])

    # Define the flatten layer
    flatten_layer = Flatten()(addition_layer)

    # Define the fully connected layer
    fc_layer = Dense(10, activation='softmax')(flatten_layer)

    # Define the model
    model = Model(inputs=input_layer, outputs=fc_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=5, batch_size=128, validation_data=(X_test, y_test))

    return model