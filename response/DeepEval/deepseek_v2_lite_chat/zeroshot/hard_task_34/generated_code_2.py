import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical


def dl_model():

    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    # Define the main path
    input_main = Input(shape=(28, 28, 1))
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_main)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Define the branch path
    input_branch = Input(shape=(28, 28, 1))
    y = Conv2D(64, kernel_size=(3, 3), activation='relu')(input_branch)

    # Concatenate features from both paths
    concat = Concatenate()([x, y])

    # Flatten and pass through a fully connected layer
    flatten = Flatten()(concat)
    output_fc = Dense(10, activation='softmax')(flatten)

    # Define the model
    model = Model(inputs=[input_main, input_branch], outputs=output_fc)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Print the model summary
    model.summary()

    # Return the constructed model
    return model