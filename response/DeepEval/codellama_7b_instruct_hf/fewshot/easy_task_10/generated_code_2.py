import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras.utils import to_categorical
from keras.optimizers import Adam

 和 return model
def dl_model():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize the pixel values to be between 0 and 1
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Convert class labels to categorical values
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Define the model architecture
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(10, activation='softmax'))

    # Compile the model with a loss function and an optimizer
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

    # Train the model on the training data
    model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))

    
    return model