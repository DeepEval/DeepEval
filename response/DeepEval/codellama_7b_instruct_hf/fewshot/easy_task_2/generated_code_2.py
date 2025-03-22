import keras
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def dl_model():
    # Create the model
    model = Sequential()

    # Add the first sequential feature extraction layer
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Add the second sequential feature extraction layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Add the third convolutional layer
    model.add(Conv2D(128, (3, 3), activation='relu'))

    # Add the fourth convolutional layer
    model.add(Conv2D(256, (3, 3), activation='relu'))

    # Add the average pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Add the flatten layer
    model.add(Flatten())

    # Add the first fully connected layer
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))

    # Add the second fully connected layer
    model.add(Dense(1000, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model