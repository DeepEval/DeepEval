import keras
from keras.datasets import mnist
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def dl_model():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Preprocess the data
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    conv1 = Conv2D(32, (1, 1), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2), padding='valid')(conv1)

    drop1 = Dropout(0.5)(pool1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(drop1)
    conv2 = Conv2D(64, (1, 1), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2), padding='valid')(conv2)

    drop2 = Dropout(0.5)(pool2)

    flat = Flatten()(drop2)

    dense1 = Dense(128, activation='relu')(flat)
    dense2 = Dense(64, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model