import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense


def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first branch
    branch1 = Conv2D(32, (1, 1), activation='relu')(input_shape)

    # Define the second branch
    branch2 = Conv2D(64, (1, 1), activation='relu')(branch1)
    branch2 = Conv2D(64, (3, 3), activation='relu')(branch2)

    # Define the third branch
    branch3 = Conv2D(128, (1, 1), activation='relu')(branch1)
    branch3 = Conv2D(128, (5, 5), activation='relu')(branch3)

    # Define the fourth branch
    branch4 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(input_shape)
    branch4 = Conv2D(64, (1, 1), activation='relu')(branch4)

    # Merge the branches
    merged = Add()([branch1, branch2, branch3, branch4])

    # Flatten the features
    flattened = Flatten()(merged)

    # Add two fully connected layers
    fc1 = Dense(128, activation='relu')(flattened)
    fc2 = Dense(10, activation='softmax')(fc1)

    # Define the model
    model = keras.Model(inputs=input_shape, outputs=fc2)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    return model