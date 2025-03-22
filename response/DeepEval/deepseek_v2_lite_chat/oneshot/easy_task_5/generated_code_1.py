import keras
from keras.datasets import mnist
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the data
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Reshape the data
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

    # Define the input shape
    input_shape = (28, 28, 1)

    # Input layer
    input_layer = Input(shape=input_shape)

    # Reduce dimensionality with 1x1 convolution
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(input_layer)

    # Extract features with 3x3 convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(conv1)

    # Restore dimensionality with another 1x1 convolution
    conv3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(conv2)

    # Flatten the output
    flatten = Flatten()(conv3)

    # Fully connected layer
    dense = Dense(units=10, activation='softmax')(flatten)

    # Model construction
    model = keras.Model(inputs=input_layer, outputs=dense)

    return model

model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()