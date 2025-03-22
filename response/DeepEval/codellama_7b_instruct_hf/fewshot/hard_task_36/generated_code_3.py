import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the model
    model = keras.Sequential([
        # Main pathway
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        keras.layers.Conv2D(64, (1, 1), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.5),
        # Branch pathway
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.Conv2D(64, (1, 1), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        # Merge the two pathways
        keras.layers.Add()
    ])

    # Flatten the output of the addition layer
    model.add(keras.layers.Flatten())

    # Add a fully connected layer
    model.add(keras.layers.Dense(128, activation='relu'))

    # Add a fully connected layer
    model.add(keras.layers.Dense(64, activation='relu'))

    # Add a fully connected layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model