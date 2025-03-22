import keras
from keras.datasets import mnist
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from keras.models import Model

def dl_model():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape data
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255.
    x_test = x_test / 255.
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Convolutional layer followed by max pooling
    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Additional convolutional layer
    conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(pool1)

    # Flatten the feature maps
    flat = Flatten()(conv2)

    # Two fully connected layers
    dense1 = Dense(128, activation='relu')(flat)
    dense2 = Dense(64, activation='relu')(dense1)

    # Output layer with softmax activation
    output_layer = Dense(10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])