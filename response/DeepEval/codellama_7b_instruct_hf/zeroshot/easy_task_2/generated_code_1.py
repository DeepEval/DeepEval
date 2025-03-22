import keras
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout, Softmax
from keras.models import Model

def dl_model():
    # Define the input shape
    input_shape = (224, 224, 3)

    # Define the first sequential feature extraction layer
    seq1 = Sequential()
    seq1.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    seq1.add(MaxPool2D((2, 2)))

    # Define the second sequential feature extraction layer
    seq2 = Sequential()
    seq2.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    seq2.add(MaxPool2D((2, 2)))

    # Define the third sequential feature extraction layer
    seq3 = Sequential()
    seq3.add(Conv2D(128, (3, 3), activation='relu', input_shape=input_shape))
    seq3.add(MaxPool2D((2, 2)))

    # Define the fourth sequential feature extraction layer
    seq4 = Sequential()
    seq4.add(Conv2D(256, (3, 3), activation='relu', input_shape=input_shape))
    seq4.add(MaxPool2D((2, 2)))

    # Define the fully connected layers
    fc1 = Dense(512, activation='relu')
    fc2 = Dense(1024, activation='relu')
    fc3 = Dense(1000, activation='softmax')

    # Define the model
    model = Model(inputs=input_shape, outputs=fc3)

    # Add the sequential layers to the model
    model.add(seq1)
    model.add(seq2)
    model.add(seq3)
    model.add(seq4)

    # Add the fully connected layers to the model
    model.add(fc1)
    model.add(fc2)
    model.add(fc3)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model