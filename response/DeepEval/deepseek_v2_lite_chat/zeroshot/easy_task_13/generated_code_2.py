import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, concatenate, Dropout


def dl_model():
    # Data preparation
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), 28, 28, 1))
    x_test = x_test.reshape((len(x_test), 28, 28, 1))

    # Model architecture
    input_layer = Input(shape=(28, 28, 1))

    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.5)(x)

    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.5)(x)

    x = Conv2D(32, (1, 3), activation='relu')(x)
    x = Conv2D(64, (1, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.5)(x)

    x = concatenate([x, input_layer])

    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)

    output_layer = Dense(10, activation='softmax')(x)

    # Model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Print the model summary
    model.summary()

    return model

    # Run the model
model = dl_model()