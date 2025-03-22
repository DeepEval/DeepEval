from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.applications.cifar10 import Cifar10

def dl_model():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = Cifar10.load_data()

    # Normalize the pixel values to be between 0 and 1
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the input layer
    input_layer = Input(shape=input_shape)

    # Define the first block
    x = input_layer
    x = Lambda(lambda x: tf.split(x, 3, axis=1))(x)
    x = Concatenate()(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Concatenate()([x, x, x])

    # Define the second block
    x = BatchNormalization()(x)
    x = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Concatenate()([x, x, x])

    # Define the fully connected layers
    x = Flatten()(x)
    x = Dense(units=128, activation='relu')(x)
    x = Dense(units=64, activation='relu')(x)
    x = Dense(units=10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=x)

    return model