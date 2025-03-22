from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Concatenate

def dl_model():
    # Define the input shape and the number of classes
    input_shape = (28, 28, 1)
    num_classes = 10

    # Define the first convolutional block
    conv1 = Conv2D(32, (3, 3), activation='relu')
    maxpool1 = MaxPooling2D((2, 2))
    block1 = Concatenate()([conv1, maxpool1])

    # Define the second convolutional block
    conv2 = Conv2D(64, (3, 3), activation='relu')
    maxpool2 = MaxPooling2D((2, 2))
    block2 = Concatenate()([conv2, maxpool2])

    # Define the third convolutional block
    conv3 = Conv2D(128, (3, 3), activation='relu')
    maxpool3 = MaxPooling2D((2, 2))
    block3 = Concatenate()([conv3, maxpool3])

    # Define the fourth convolutional block
    conv4 = Conv2D(256, (3, 3), activation='relu')
    maxpool4 = MaxPooling2D((2, 2))
    block4 = Concatenate()([conv4, maxpool4])

    # Flatten the feature maps and add a dense layer
    flatten = Flatten()
    dense1 = Dense(128, activation='relu')
    dense2 = Dense(64, activation='relu')
    dense3 = Dense(num_classes, activation='softmax')

    # Define the model architecture
    model = Model(inputs=input_shape, outputs=dense3)
    model.add(conv1)
    model.add(maxpool1)
    model.add(block1)
    model.add(conv2)
    model.add(maxpool2)
    model.add(block2)
    model.add(conv3)
    model.add(maxpool3)
    model.add(block3)
    model.add(conv4)
    model.add(maxpool4)
    model.add(block4)
    model.add(flatten)
    model.add(dense1)
    model.add(dense2)
    model.add(dense3)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)