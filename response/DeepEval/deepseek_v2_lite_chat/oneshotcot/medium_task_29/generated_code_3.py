import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

def dl_model():
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Add channels dimension
    x_train = x_train.reshape((x_train.shape[0], 3, 32, 32))
    x_test = x_test.reshape((x_test.shape[0], 3, 32, 32))

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Three pooling layers with varying window sizes
    def pool_layers(input_tensor):
        # 1x1 max pooling
        pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_tensor)
        # 2x2 max pooling
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(pool1)
        # 4x4 max pooling
        pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(pool2)

        # Flatten the tensors and concatenate them
        concatenated = Concatenate()(pool3)

        return concatenated

    # Construct the model
    model = Model(inputs=input_layer, outputs=pool_layers(input_tensor=input_layer))

    # Add dense layers
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))

    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()