import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense

def dl_model():
    # Main path
    main_path = Sequential()
    main_path.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    main_path.add(Conv2D(64, (3, 3), activation='relu'))
    main_path.add(Conv2D(64, (3, 3), activation='relu'))
    main_path.add(MaxPooling2D((2, 2)))

    # Branch path
    branch_path = Sequential()
    branch_path.add(GlobalAveragePooling2D())
    branch_path.add(Dense(128, activation='relu'))
    branch_path.add(Dense(64, activation='relu'))

    # Model
    model = Sequential()
    model.add(main_path)
    model.add(branch_path)
    model.add(Concatenate())
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    return model