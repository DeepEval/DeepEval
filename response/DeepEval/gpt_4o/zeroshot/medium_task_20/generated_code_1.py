from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from keras.models import Model
from keras.datasets import cifar10
from keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First path: 1x1 convolution
    path1 = Conv2D(64, (1, 1), activation='relu', padding='same')(input_layer)

    # Second path: 1x1 followed by two 3x3 convolutions
    path2 = Conv2D(64, (1, 1), activation='relu', padding='same')(input_layer)
    path2 = Conv2D(64, (3, 3), activation='relu', padding='same')(path2)
    path2 = Conv2D(64, (3, 3), activation='relu', padding='same')(path2)

    # Third path: 1x1 followed by a single 3x3 convolution
    path3 = Conv2D(64, (1, 1), activation='relu', padding='same')(input_layer)
    path3 = Conv2D(64, (3, 3), activation='relu', padding='same')(path3)

    # Fourth path: Max pooling followed by 1x1 convolution
    path4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_layer)
    path4 = Conv2D(64, (1, 1), activation='relu', padding='same')(path4)

    # Concatenate the outputs from the four paths
    concatenated = concatenate([path1, path2, path3, path4], axis=-1)

    # Flatten the concatenated outputs
    flat = Flatten()(concatenated)

    # Dense layer with 128 units
    dense = Dense(128, activation='relu')(flat)

    # Output layer with softmax activation for 10 categories
    output_layer = Dense(10, activation='softmax')(dense)

    # Model definition
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage:
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Train the model (this is just an example and can take a long time depending on the hardware)
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))