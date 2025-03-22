import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Concatenate, Flatten, Dense
from keras.datasets import cifar10
from keras.utils import to_categorical


def dl_model():

    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Convert labels to categorical
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Number of classes
    num_classes = y_train.shape[1]

    # Model inputs
    input_branch = Input(shape=(32, 32, 3))
    input_branch_conv = Conv2D(32, (1, 1), padding='same')(input_branch)
    input_branch_conv = Conv2D(64, (1, 7), padding='same')(input_branch_conv)
    input_branch_conv = Conv2D(64, (7, 1), padding='same')(input_branch_conv)

    # Feature extraction path
    path1 = input_branch_conv

    # Second path of convolutions
    input_branch_conv2 = Conv2D(32, (1, 1), padding='same')(input_branch)
    input_branch_conv2 = Conv2D(64, (1, 7), padding='same')(input_branch_conv2)
    input_branch_conv2 = Conv2D(64, (7, 1), padding='same')(input_branch_conv2)
    path2 = Add()([input_branch_conv, input_branch_conv2])

    # Concatenate the outputs
    concat = Concatenate()([path1, path2])

    # Align output dimensions
    output = Conv2D(64, (1, 1), padding='same')(concat)

    # Fully connected layers for classification
    fc = Flatten()(output)
    fc = Dense(512, activation='relu')(fc)
    fc = Dense(num_classes, activation='softmax')(fc)

    # Connect inputs and outputs
    model = Model(inputs=[input_branch, input_branch_conv2], outputs=[fc])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Return the model
    return model

# Example usage:
model = dl_model()
model.fit([x_train, x_train_conv], y_train, validation_data=([x_test, x_test_conv], y_test))