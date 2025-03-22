import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dense
from keras.applications import VGG16

def dl_model():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize the pixel values to be between 0 and 1
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first max pooling layer with a 1x1 window size
    pooling1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))

    # Define the second max pooling layer with a 2x2 window size
    pooling2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

    # Define the third max pooling layer with a 4x4 window size
    pooling3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))

    # Define the first fully connected layer
    dense1 = Dense(units=128, activation='relu')

    # Define the second fully connected layer
    dense2 = Dense(units=64, activation='relu')

    # Define the output layer
    output_layer = Dense(units=10, activation='softmax')

    # Define the input layer
    input_layer = Input(shape=input_shape)

    # Define the first convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Define the second convolutional layer
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Define the third convolutional layer
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)

    # Define the fourth convolutional layer
    conv4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)

    # Define the first max pooling layer
    pooling1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(conv1)

    # Define the second max pooling layer
    pooling2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    # Define the third max pooling layer
    pooling3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(conv3)

    # Define the fourth max pooling layer
    pooling4 = MaxPooling2D(pool_size=(8, 8), strides=(8, 8))(conv4)

    # Define the flatten layer
    flatten = Flatten()(pooling4)

    # Define the first fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten)

    # Define the second fully connected layer
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Define the output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model