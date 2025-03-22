import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate


def dl_model():
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize the pixel values
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Define the branch path
    input_branch = Input(shape=(32, 32, 3))
    x = Conv2D(32, (3, 3), activation='relu')(input_branch)
    x = MaxPooling2D()(x)

    # Define the main path
    input_main = Input(shape=(32, 32, 3))
    x = Conv2D(32, (3, 3), activation='relu')(input_main)
    x = MaxPooling2D()(x)

    # Concatenate the branch and main paths
    x = concatenate([x, x])

    # Add more layers to the combined path
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D()(x)

    # Flatten and project onto 10 classes
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    output = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=[input_branch, input_main], outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Print the model summary
    model.summary()

    return model


model.fit([x_train_branch, x_train_main], y_train, validation_data=([x_test_branch, x_test_main], y_test))