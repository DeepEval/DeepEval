from keras.layers import Input, Conv2D, AveragePooling2D, Dropout, GlobalAveragePooling2D, Dense, Flatten
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical
import keras

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # First specialized block
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = Conv2D(32, (1, 1), activation='relu', padding='same')(x)
    x = Conv2D(32, (1, 1), activation='relu', padding='same')(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # Second specialized block
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (1, 1), activation='relu', padding='same')(x)
    x = Conv2D(64, (1, 1), activation='relu', padding='same')(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # Global average pooling layer
    x = GlobalAveragePooling2D()(x)

    # Flattening layer is not necessary after GlobalAveragePooling2D since it already flattens
    # A fully connected layer
    output_layer = Dense(10, activation='softmax')(x)

    # Creating the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Example usage:
# Load the MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Instantiate and train the model
model = dl_model()
model.summary()
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))