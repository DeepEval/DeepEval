import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate


def dl_model():
    
    cifar10.load_data(batch_size=64)

    # Data preprocessing
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    # Model parameters
    input_shape = (32, 32, 3)  # Adjusted for CIFAR-10
    num_classes = 10

    # Block 1: Global average pooling, two fully connected layers
    input_layer = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D()(x)
    x = Flatten()(x)

    fc1 = Dense(512, activation='relu')(x)
    fc2 = Dense(num_classes)(fc1)

    # Block 2: Two 3x3 convolutional layers, max pooling
    block2_input = Input(shape=(32, 32, 3))
    x = Conv2D(64, (3, 3), activation='relu')(block2_input)
    x = MaxPooling2D()(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D()(x)

    # Branch from Block 1 to Block 2
    branch_output = Flatten()(x)

    # Fusion
    z = concatenate([fc2, branch_output])

    # Classification layers
    z = Dense(512, activation='relu')(z)
    output = Dense(num_classes, activation='softmax')(z)

    # Model
    model = Model(inputs=[input_layer, block2_input], outputs=[output])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

    # Instantiate and return the model
    model = dl_model()