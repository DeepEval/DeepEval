from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the model using Functional API
def dl_model():
    # Input layer
    inputs = keras.Input(shape=(32, 32, 3))

    # Convolutional layers with channel concatenation
    conv_1 = Convolution2D(32, (3, 3), padding='same')(inputs)
    conv_1 = Activation('relu')(conv_1)
    conv_2 = Convolution2D(32, (3, 3), padding='same')(conv_1)
    conv_2 = Activation('relu')(conv_2)
    conv_3 = Convolution2D(64, (3, 3), padding='same')(conv_2)
    conv_3 = Activation('relu')(conv_3)

    # Pooling layers
    pool_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)
    pool_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)
    pool_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)

    # Flatten the output from the convolutional layers
    flatten = Flatten()(pool_3)

    # Fully connected layers
    dense_1 = Dense(256, activation='relu')(flatten)
    dense_2 = Dense(10, activation='softmax')(dense_1)

    # Create the model
    model = keras.Model(inputs, dense_2)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Train the model
model = dl_model()
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])