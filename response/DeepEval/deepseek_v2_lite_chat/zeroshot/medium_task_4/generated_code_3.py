from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator


def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # One-hot encode the labels
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    # Define paths for the two pathways
    input_path1 = Input(shape=(32, 32, 3))
    input_path2 = Input(shape=(32, 32, 3))

    # Path 1: two convolution blocks followed by average pooling
    conv_block1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_path1)
    conv_block2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv_block1)
    avg_pool1 = MaxPooling2D(pool_size=(2, 2))(conv_block2)

    # Flatten and fully connected layer for path 1
    flatten1 = Flatten()(avg_pool1)
    fully_connected1 = Dense(128, activation='relu')(flatten1)

    # Path 2: single convolution layer
    conv_path2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(input_path2)
    avg_pool2 = MaxPooling2D(pool_size=(2, 2))(conv_path2)

    # Concatenate the features from both paths and add a fully connected layer
    concat = concatenate([fully_connected1, avg_pool2])
    fully_connected2 = Dense(128, activation='relu')(concat)

    # Probability distribution over 10 classes using a fully connected layer
    output = Dense(10, activation='softmax')(fully_connected2)

    # Create the model
    model = Model(inputs=[input_path1, input_path2], outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model