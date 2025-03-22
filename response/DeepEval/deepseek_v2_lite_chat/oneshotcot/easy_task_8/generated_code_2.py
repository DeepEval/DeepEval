import keras
from keras.datasets import mnist
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def dl_model():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape data
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

    # Normalize data
    x_train /= 255
    x_test /= 255

    # Convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    # Construct model
    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    drop1 = Dropout(0.5)(pool2)
    drop2 = Dropout(0.5)(drop1)

    flat = Flatten()(drop2)
    dense1 = Dense(units=128, activation='relu')(flat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model