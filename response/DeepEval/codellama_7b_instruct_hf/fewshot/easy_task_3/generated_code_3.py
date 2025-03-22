import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

和return model
def dl_model():
    # First convolutional block
    conv1 = Conv2D(32, (3, 3), activation='relu')
    pool1 = MaxPooling2D((2, 2))

    # Second convolutional block
    conv2 = Conv2D(64, (3, 3), activation='relu')
    pool2 = MaxPooling2D((2, 2))

    # Third convolutional block
    conv3 = Conv2D(128, (3, 3), activation='relu')
    pool3 = MaxPooling2D((2, 2))

    # Fourth convolutional block
    conv4 = Conv2D(256, (3, 3), activation='relu')
    pool4 = MaxPooling2D((2, 2))

    # Flatten layer
    flatten = Flatten()

    # First fully connected layer
    fc1 = Dense(128, activation='relu')

    # Second fully connected layer
    fc2 = Dense(64, activation='relu')

    # Output layer
    output = Dense(10, activation='softmax')

    # Define the model
    model = Model(inputs=input_layer, outputs=output)
    model.add(conv1)
    model.add(pool1)
    model.add(conv2)
    model.add(pool2)
    model.add(conv3)
    model.add(pool3)
    model.add(conv4)
    model.add(pool4)
    model.add(flatten)
    model.add(fc1)
    model.add(fc2)
    model.add(output)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

    
    return model