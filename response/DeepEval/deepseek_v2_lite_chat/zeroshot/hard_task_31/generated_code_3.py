import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Add, SeparableConv2D, Lambda, Flatten, Dense
from tensorflow.keras.models import Model


def dl_model():

    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # One-hot encode the labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Data augmentation
    datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    datagen.fit(x_train)

    # Define the main path of the model
    input_main = Input(shape=(32, 32, 3))
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_main)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    dropout_main = Dropout(0.5)(x)

    restored_x = Conv2D(32, (1, 1), activation='sigmoid', padding='same')(dropout_main)
    x = Add()([x, restored_x])

    # Define the branch path of the model
    branch_input = Input(shape=(32, 32, 3))
    x_branch = branch_input

    # Separate and process each group
    for i in range(3):
        x = SeparableConv2D(64 * (2 ** (i + 1)), (1, 1), activation='relu', padding='same')(x_branch)
        x = SeparableConv2D(64 * (2 ** (i + 1)), (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.5)(x)

    x_group1 = Flatten()(x)
    x_group2 = Flatten()(x_branch)
    x_group3 = Flatten()(x)

    concat_x = Concatenate()([x_group1, x_group2, x_group3])

    # Combine the outputs from both paths
    output_main = Dense(10, activation='softmax')(Flatten(name="flatten_main")(x))
    output_branch = Dense(10, activation='softmax')(Flatten(name="flatten_branch")(x_branch))

    # Create the model
    model = Model(inputs=[input_main, branch_input], outputs=[output_main, output_branch])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Summary of the model
    model.summary()

    # Return the model
    return model

# Call the function and print the model
model = dl_model()