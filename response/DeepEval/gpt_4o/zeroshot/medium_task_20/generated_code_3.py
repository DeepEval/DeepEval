from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from keras.models import Model
from keras.datasets import cifar10
from keras.utils import to_categorical

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Path 1: 1x1 Convolution
    path1 = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)

    # Path 2: 1x1 Convolution followed by two 3x3 Convolutions
    path2 = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)
    path2 = Conv2D(32, (3, 3), activation='relu', padding='same')(path2)
    path2 = Conv2D(32, (3, 3), activation='relu', padding='same')(path2)

    # Path 3: 1x1 Convolution followed by a 3x3 Convolution
    path3 = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)
    path3 = Conv2D(32, (3, 3), activation='relu', padding='same')(path3)

    # Path 4: Max Pooling followed by a 1x1 Convolution
    path4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_layer)
    path4 = Conv2D(32, (1, 1), activation='relu', padding='same')(path4)

    # Concatenate all paths
    concatenated = Concatenate()([path1, path2, path3, path4])

    # Flatten the concatenated output
    flattened = Flatten()(concatenated)

    # Dense layer with 128 units
    dense = Dense(128, activation='relu')(flattened)

    # Output layer with softmax activation for 10 classes
    output_layer = Dense(10, activation='softmax')(dense)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Load CIFAR-10 data and preprocess it
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Get the model
model = dl_model()

# Summary of the model
model.summary()

# Fit the model
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))