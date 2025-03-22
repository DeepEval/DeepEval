import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Conv2DTranspose, AveragePooling2D
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.datasets import cifar10
from keras.utils import to_categorical


def dl_model():
    # Load and preprocess CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Define the main path and branch paths
    input_main = Input(shape=(32, 32, 3))
    input_branch = Input(shape=(32, 32, 3))

    # Main path layers
    x = Conv2D(64, (1, 1), padding='same')(input_main)
    x_branch = input_branch

    x = Conv2D(64, (3, 3), padding='same')(x)
    x_branch = Conv2D(64, (3, 3), padding='same')(x_branch)

    x = AveragePooling2D()(x)
    x_branch = AveragePooling2D()(x_branch)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x_branch = Conv2DTranspose(128, (3, 3), strides=2, padding='same')(x_branch)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x_branch = Conv2DTranspose(128, (3, 3), strides=2, padding='same')(x_branch)

    # Concatenate main path and branch path outputs
    x = Concatenate()([x, x_branch])

    # Branch path layers
    x = Conv2D(64, (1, 1), padding='same')(input_branch)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = UpSampling2D(size=2)(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = UpSampling2D(size=2)(x)

    # Output layers
    x = Conv2D(10, (1, 1), padding='same')(x)
    x = Flatten()(x)
    output = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=[input_main, input_branch], outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Return the constructed model
    return model