import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.applications.cifar10 import CIFAR10
from keras.preprocessing.image import ImageDataGenerator

# Load the CIFAR-10 dataset
cifar10 = CIFAR10(train_batch_size=32, test_batch_size=32)


def dl_model():
    # Define the input shape for the model
    input_shape = (32, 32, 3)

    # Define the first convolutional layer with a kernel size of (3, 3) and a stride of (1, 1)
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')

    # Define the first max pooling layer with a pool size of (2, 2) and a stride of (2, 2)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)

    # Define three more convolutional layers with kernel sizes of (3, 3), (3, 3), and (3, 3) respectively
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')
    conv4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')

    # Define the first max pooling layer with a pool size of (1, 1) and a stride of (1, 1) for all of them
    max_pooling2 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(conv2)
    max_pooling3 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(conv3)
    max_pooling4 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(conv4)

    # Concatenate the output of all the pooling layers
    output = Concatenate()([max_pooling, max_pooling2, max_pooling3, max_pooling4])

    # Apply batch normalization to the concatenated output
    bath_norm = BatchNormalization()(output)

    # Flatten the output of the batch normalization layer
    flatten_layer = Flatten()(bath_norm)

    # Define two fully connected layers with 128 and 64 units respectively
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Define the output layer with 10 units, which represents the 10 classes in the CIFAR-10 dataset
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Define the model
    model = keras.Model(inputs=input_shape, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Return the model
    return model