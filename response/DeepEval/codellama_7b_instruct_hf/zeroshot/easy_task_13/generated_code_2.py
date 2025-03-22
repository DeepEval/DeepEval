from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense


def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the first 1x1 convolutional layer
    conv1 = Conv2D(32, (1, 1), activation='relu')(input_shape)

    # Define the second 1x1 convolutional layer
    conv2 = Conv2D(32, (1, 1), activation='relu')(conv1)

    # Define the third 3x1 convolutional layer
    conv3 = Conv2D(64, (3, 1), activation='relu')(conv2)

    # Define the fourth 1x3 convolutional layer
    conv4 = Conv2D(64, (1, 3), activation='relu')(conv3)

    # Define the 1x1 convolutional layer to restore the number of channels
    conv5 = Conv2D(1, (1, 1), activation='relu')(conv4)

    # Define the dropout layers
    drop1 = Dropout(0.2)(conv5)
    drop2 = Dropout(0.2)(drop1)

    # Define the flattening layer
    flatten = Flatten()(drop2)

    # Define the fully connected layer
    fc = Dense(10, activation='softmax')(flatten)

    # Define the model
    model = Model(inputs=input_shape, outputs=fc)

    # Compile the model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])


    # Train the model on the MNIST dataset
    model.fit(x, y, batch_size=32, epochs=10, validation_split=0.2)

    return model