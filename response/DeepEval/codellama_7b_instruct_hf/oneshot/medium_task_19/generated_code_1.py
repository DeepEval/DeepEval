from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model
from keras.applications.cifar10 import Cifar10


def dl_model():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = Cifar10().load_data()

    # Normalize the inputs
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the model
    input_layer = Input(shape=input_shape)

    # First branch
    x = Conv2D(32, (1, 1), strides=(2, 2), padding='valid', activation='relu')(input_layer)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
    x = Conv2D(64, (1, 1), strides=(2, 2), padding='valid', activation='relu')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(64, (5, 5), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
    x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)

    # Second branch
    x = Conv2D(32, (1, 1), strides=(2, 2), padding='valid', activation='relu')(input_layer)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
    x = Conv2D(64, (1, 1), strides=(2, 2), padding='valid', activation='relu')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(64, (5, 5), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
    x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)

    # Third branch
    x = Conv2D(32, (1, 1), strides=(2, 2), padding='valid', activation='relu')(input_layer)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
    x = Conv2D(64, (1, 1), strides=(2, 2), padding='valid', activation='relu')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(64, (5, 5), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
    x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)

    # Fourth branch
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
    x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(64, (5, 5), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
    x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)

    # Concatenate the outputs of the branches
    x = Concatenate()([x, x, x, x])

    # Fully connected layers
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=input_layer, outputs=x)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

    return model