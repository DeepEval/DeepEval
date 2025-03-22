import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def dl_model():

    # Load the CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize the pixel values to be between 0 and 1
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first parallel convolutional branch
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)

    # Define the second parallel convolutional branch
    conv2 = Conv2D(32, (5, 5), activation='relu')(input_layer)

    # Define the addition layer
    addition = Add()([conv1, conv2])

    # Define the global average pooling layer
    pooling = GlobalAveragePooling2D()(addition)

    # Define the first fully connected layer
    fc1 = Dense(64, activation='relu')(pooling)

    # Define the second fully connected layer
    fc2 = Dense(10, activation='softmax')(fc1)

    # Define the model
    model = Model(inputs=input_layer, outputs=fc2)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # Evaluate the model
    model.evaluate(X_test, y_test)

    return model