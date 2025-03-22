import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model


def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_shape)

    # Define the second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Define the third convolutional layer
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)

    # Define the first fully connected layer
    fc1 = Dense(units=256, activation='relu')(conv3)

    # Define the second fully connected layer
    fc2 = Dense(units=10, activation='softmax')(fc1)

    # Define the model
    model = Model(inputs=input_shape, outputs=fc2)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

    return model