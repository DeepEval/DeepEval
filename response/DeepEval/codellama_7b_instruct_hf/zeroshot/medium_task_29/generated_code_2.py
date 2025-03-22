import keras
from keras.models import Model
from keras.layers import Input, MaxPooling2D, Flatten, Dense
from keras.applications.cifar10 import Cifar10

def dl_model():
    # Load the CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = Cifar10.load_data()

    # Normalize the pixel values to be between 0 and 1
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first max pooling layer with a window size of 1x1
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(input_shape)

    # Define the second max pooling layer with a window size of 2x2
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(pool1)

    # Define the third max pooling layer with a window size of 4x4
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(pool2)

    # Flatten the output of the third pooling layer
    flattened = Flatten()(pool3)

    # Concatenate the flattened outputs of the first three pooling layers
    concatenated = keras.layers.concatenate([flattened, flattened, flattened])

    # Define the first fully connected layer with 128 units
    dense1 = Dense(128, activation='relu')(concatenated)

    # Define the second fully connected layer with 10 units
    dense2 = Dense(10, activation='softmax')(dense1)

    # Define the model
    model = Model(inputs=input_shape, outputs=dense2)

    # Compile the model with a loss function and an optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Return the constructed model
    return model