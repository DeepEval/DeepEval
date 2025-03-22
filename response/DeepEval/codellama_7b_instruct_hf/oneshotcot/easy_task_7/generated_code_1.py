import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.datasets import mnist


def dl_model():
    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize pixel values to be between 0 and 1
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    # Define input layer
    input_layer = Input(shape=(28, 28, 1))

    # Define main path
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)

    # Define branch path
    branch_path = Input(shape=(28, 28, 1))

    # Combine outputs from both paths through an addition operation
    output = Concatenate()([main_path, branch_path])

    # Apply batch normalization and flatten the output
    output = BatchNormalization()(output)
    output = Flatten()(output)

    # Apply fully connected layers to produce final classification probabilities
    output = Dense(units=128, activation='relu')(output)
    output = Dense(units=10, activation='softmax')(output)

    # Define model
    model = keras.Model(inputs=input_layer, outputs=output)

    # Compile model with a loss function and an optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train model on training data
    model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test))

    # Return the constructed model
    return model