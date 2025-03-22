import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Activation

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the main path
    main_path = Sequential()
    main_path.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    main_path.add(MaxPooling2D((2, 2)))
    main_path.add(Conv2D(64, (3, 3), activation='relu'))
    main_path.add(MaxPooling2D((2, 2)))
    main_path.add(Conv2D(128, (3, 3), activation='relu'))
    main_path.add(MaxPooling2D((2, 2)))
    main_path.add(Flatten())
    main_path.add(Dense(10, activation='softmax'))

    # Define the branch path
    branch_path = Sequential()
    branch_path.add(Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
    branch_path.add(MaxPooling2D((2, 2)))
    branch_path.add(Conv2D(32, (3, 3), activation='relu'))
    branch_path.add(MaxPooling2D((2, 2)))
    branch_path.add(Conv2D(64, (3, 3), activation='relu'))
    branch_path.add(MaxPooling2D((2, 2)))
    branch_path.add(Flatten())
    branch_path.add(Dense(10, activation='softmax'))

    # Define the model
    model = Sequential()
    model.add(main_path)
    model.add(branch_path)
    model.add(Add())
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model