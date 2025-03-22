from keras.models import Model
from keras.layers import Input, Flatten, Concatenate, Dense, Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization
from keras.utils import to_categorical
from keras.applications import VGG16


def dl_model():
    
    # Load the MNIST dataset
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize the pixel values to be between 0 and 1
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    # Convert class labels to categorical values
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the first block
    inputs = Input(shape=input_shape)
    x = BatchNormalization()(inputs)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Define the second block
    inputs = Input(shape=input_shape)
    x = BatchNormalization()(inputs)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Concatenate the outputs of the two blocks
    outputs = Concatenate()([x, x])

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test))

    return model