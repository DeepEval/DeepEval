import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Flatten, Reshape

 和 return model
def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first path (main path)
    main_path = Sequential()
    main_path.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    main_path.add(Conv2D(64, (3, 3), activation='relu'))
    main_path.add(MaxPooling2D((2, 2)))

    # Define the second path (branch path)
    branch_path = Sequential()
    branch_path.add(GlobalAveragePooling2D())
    branch_path.add(Dense(128, activation='relu'))
    branch_path.add(Dense(64, activation='relu'))

    # Define the third path (additional path)
    additional_path = Sequential()
    additional_path.add(Flatten())
    additional_path.add(Dense(128, activation='relu'))
    additional_path.add(Dense(64, activation='relu'))

    # Define the model
    model = Sequential()
    model.add(main_path)
    model.add(branch_path)
    model.add(additional_path)
    model.add(Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    return model