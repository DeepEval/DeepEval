from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Model

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the first 1x1 convolutional layer
    conv1 = Conv2D(32, (1, 1), activation='relu')

    # Define the second 1x1 convolutional layer
    conv2 = Conv2D(64, (1, 1), activation='relu')

    # Define the third 3x1 convolutional layer
    conv3 = Conv2D(64, (3, 1), activation='relu')

    # Define the fourth 1x3 convolutional layer
    conv4 = Conv2D(64, (1, 3), activation='relu')

    # Define the dropout layers
    dropout1 = Dropout(0.25)
    dropout2 = Dropout(0.5)

    # Define the flattening layer
    flatten = Flatten()

    # Define the fully connected layer
    dense = Dense(10, activation='softmax')

    # Define the input layer
    input_layer = Input(shape=input_shape)

    # Define the output layer
    output_layer = dense(flatten(dropout2(conv4(dropout1(conv3(conv2(conv1(input_layer))))))))

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on the MNIST dataset
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))