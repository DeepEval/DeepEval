import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize the pixel values to be between 0 and 1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # Define the input shape
    input_shape = (32, 32, 3)

    # Input layer
    input_layer = Input(shape=input_shape)

    # Convolutional layer 1
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)

    # Convolutional layer 2
    conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)

    # Max-pooling layer
    pool = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Concatenate the outputs of the convolutional paths
    concatenated = Concatenate()([pool, conv2])

    # Batch normalization
    bn = BatchNormalization()(concatenated)

    # Flatten the output tensor
    flat = Flatten()(bn)

    # Fully connected layer 1
    dense1 = Dense(512, activation='relu')(flat)

    # Fully connected layer 2
    dense2 = Dense(256, activation='relu')(dense1)

    # Output layer
    output_layer = Dense(10, activation='softmax')(dense2)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model