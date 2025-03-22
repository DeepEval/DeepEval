import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.layers.core import Lambda
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16


def dl_model():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize the data
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Convert the labels to categorical
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Define the model architecture
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    x1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x1 = Conv2D(32, (3, 3), activation='relu')(x1)
    x1 = Conv2D(32, (3, 3), activation='relu')(x1)
    x1 = MaxPooling2D(pool_size=(2, 2))(x1)

    # Block 2
    x2 = Conv2D(64, (3, 3), activation='relu')(input_layer)
    x2 = Conv2D(64, (3, 3), activation='relu')(x2)
    x2 = Conv2D(64, (3, 3), activation='relu')(x2)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)

    # Block 3
    x3 = Conv2D(128, (3, 3), activation='relu')(input_layer)
    x3 = Conv2D(128, (3, 3), activation='relu')(x3)
    x3 = Conv2D(128, (3, 3), activation='relu')(x3)
    x3 = MaxPooling2D(pool_size=(2, 2))(x3)

    # Concatenate the outputs of the blocks
    x = Concatenate()([x1, x2, x3])

    # Batch normalization
    x = BatchNormalization()(x)

    # Flatten the output
    x = Flatten()(x)

    # Fully connected layers
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    # Output layer
    output = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=input_layer, outputs=output)

    # Compile the model
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

    return model