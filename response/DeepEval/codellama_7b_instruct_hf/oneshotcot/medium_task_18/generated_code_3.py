import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.applications import VGG16


def dl_model():

    # Load the pre-trained VGG16 model
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the first convolutional block
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Define the second convolutional block
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Define the third convolutional block
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv2)

    # Define the fourth convolutional block
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)

    # Define the fifth convolutional block
    conv5 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv4)

    # Define the max pooling layer
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv5)

    # Define the flatten layer
    flatten = Flatten()(max_pooling)

    # Define the first fully connected layer
    fc1 = Dense(units=128, activation='relu')(flatten)

    # Define the second fully connected layer
    fc2 = Dense(units=64, activation='relu')(fc1)

    # Define the output layer
    output = Dense(units=10, activation='softmax')(fc2)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    # Train the model
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test loss: {loss:.3f}, Test accuracy: {accuracy:.3f}')

    return model