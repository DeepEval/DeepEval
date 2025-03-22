import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Concatenate, Dense, Reshape
from keras.datasets import mnist
from keras.utils import to_categorical


def dl_model():

    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    # Define the first block
    input_layer = Input(shape=(28, 28, 1))
    x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(x)
    x = Flatten()(x)

    # Define the second block
    input_branch1 = Conv2D(64, kernel_size=(1, 1), activation='relu', padding='same')(input_layer)
    input_branch2 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    input_branch3 = Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same')(input_layer)
    input_branch4 = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same')(input_layer)

    x = Concatenate()([input_branch1, input_branch2, input_branch3, input_branch4])
    x = Flatten()(x)

    # Fully connected layers
    x = Dense(256, activation='relu')(x)
    output = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=input_layer, outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Print model summary
    model.summary()

    return model