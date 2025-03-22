# Import necessary packages
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, concatenate, multiply
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10

def dl_model():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the inputs
    input_branch_1 = Input(shape=input_shape)
    input_branch_2 = Input(shape=input_shape)

    # Define the first convolutional branch with 3x3 kernel
    x_branch_1 = Conv2D(32, (3, 3), padding='same')(input_branch_1)
    x_branch_1 = BatchNormalization()(x_branch_1)
    x_branch_1 = Activation('relu')(x_branch_1)
    x_branch_1 = MaxPooling2D((2, 2))(x_branch_1)

    # Define the second convolutional branch with 5x5 kernel
    x_branch_2 = Conv2D(32, (5, 5), padding='same')(input_branch_2)
    x_branch_2 = BatchNormalization()(x_branch_2)
    x_branch_2 = Activation('relu')(x_branch_2)
    x_branch_2 = MaxPooling2D((2, 2))(x_branch_2)

    # Combine the outputs of the two branches through addition
    combined_output = concatenate([x_branch_1, x_branch_2])

    # Apply global average pooling to compress the features
    compressed_output = GlobalAveragePooling2D()(combined_output)

    # Define the first fully connected layer
    x = Dense(128, activation='relu')(compressed_output)
    x = BatchNormalization()(x)

    # Define the second fully connected layer that applies a softmax function
    x = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=[input_branch_1, input_branch_2], outputs=x)

    return model

model = dl_model()
model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])