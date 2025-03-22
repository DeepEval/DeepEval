import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.applications.vgg16 import VGG16

def dl_model():
    # Create the base model using VGG16
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    # Create the first sequential block
    first_block = Sequential()
    first_block.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    first_block.add(Conv2D(32, (3, 3), activation='relu'))
    first_block.add(MaxPooling2D((2, 2)))

    # Create the second sequential block
    second_block = Sequential()
    second_block.add(Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    second_block.add(Conv2D(64, (3, 3), activation='relu'))
    second_block.add(MaxPooling2D((2, 2)))
    second_block.add(Flatten())
    second_block.add(Dense(128, activation='relu'))
    second_block.add(Dense(10, activation='softmax'))

    # Create the model by combining the two sequential blocks
    model = Sequential()
    model.add(first_block)
    model.add(second_block)

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model