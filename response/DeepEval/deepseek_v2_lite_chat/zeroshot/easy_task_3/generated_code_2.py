from keras.datasets import mnist
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

def dl_model():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the images to the [0, 1] range
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Reshape the datasets to have a single color channel (grayscale)
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

    # Define the input shape (samples, rows, columns, channels)
    input_shape = (28, 28, 1)

    # Define the model inputs and outputs
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Get the model
model = dl_model()
model.summary()