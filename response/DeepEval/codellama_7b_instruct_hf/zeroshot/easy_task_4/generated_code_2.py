import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    # Define the first sequential block
    first_block = Sequential()
    first_block.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    first_block.add(MaxPooling2D((2, 2)))
    first_block.add(Conv2D(64, (3, 3), activation='relu'))
    first_block.add(MaxPooling2D((2, 2)))
    first_block.add(Conv2D(64, (3, 3), activation='relu'))
    first_block.add(MaxPooling2D((2, 2)))

    # Define the second sequential block
    second_block = Sequential()
    second_block.add(Conv2D(128, (3, 3), activation='relu', input_shape=(14, 14, 64)))
    second_block.add(MaxPooling2D((2, 2)))
    second_block.add(Conv2D(128, (3, 3), activation='relu'))
    second_block.add(MaxPooling2D((2, 2)))
    second_block.add(Conv2D(128, (3, 3), activation='relu'))
    second_block.add(MaxPooling2D((2, 2)))

    # Flatten the feature maps
    flatten = Flatten()(first_block.output)
    flatten = Flatten()(second_block.output)

    # Define the fully connected layers
    fc1 = Dense(512, activation='relu')(flatten)
    fc2 = Dense(10, activation='softmax')(fc1)

    # Define the model
    model = Sequential()
    model.add(first_block)
    model.add(second_block)
    model.add(flatten)
    model.add(fc1)
    model.add(fc2)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model