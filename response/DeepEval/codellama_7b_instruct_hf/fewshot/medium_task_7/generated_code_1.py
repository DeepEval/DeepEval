import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense


def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first convolutional layer
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_shape)

    # Define the second convolutional layer
    conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)

    # Define the third convolutional layer
    conv3 = Conv2D(128, (3, 3), activation='relu')(conv2)

    # Define the fourth convolutional layer
    conv4 = Conv2D(256, (3, 3), activation='relu')(conv3)

    # Define the first max pooling layer
    max_pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Define the second max pooling layer
    max_pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Define the third max pooling layer
    max_pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Define the fourth max pooling layer
    max_pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Define the addition layer
    add_layer = Add()([max_pool1, max_pool2, max_pool3, max_pool4])

    # Define the flatten layer
    flatten_layer = Flatten()(add_layer)

    # Define the first fully connected layer
    dense1 = Dense(64, activation='relu')(flatten_layer)

    # Define the second fully connected layer
    dense2 = Dense(10, activation='softmax')(dense1)

    # Define the model
    model = keras.Model(inputs=input_shape, outputs=dense2)

    # Compile the model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    return model