import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout, Flatten, Dense


def dl_model():
    
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize the pixel values
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Split the channel dimension into three groups
    x_train_split = tf.split(x_train, 3, axis=3)
    x_test_split = tf.split(x_test, 3, axis=3)

    # Define the main pathway layers
    input_main = Input(shape=(32, 32, 3))
    conv1 = Conv2D(32, (1, 1))(input_main)
    conv2 = Conv2D(32, (1, 1))(conv1)

    x1 = Conv2D(32, (3, 3), padding='same')(x_train_split[0])
    x1 = Conv2D(64, (3, 3), padding='same')(x1)
    x1 = Dropout(0.2)(x1)

    x2 = Conv2D(32, (3, 3), padding='same')(x_train_split[1])
    x2 = Conv2D(64, (3, 3), padding='same')(x2)
    x2 = Dropout(0.2)(x2)

    x3 = Conv2D(32, (3, 3), padding='same')(x_train_split[2])
    x3 = Conv2D(64, (3, 3), padding='same')(x3)
    x3 = Dropout(0.2)(x3)

    # Concatenate the outputs from the three groups for the main pathway
    x_main = Concatenate()([x1, x2, x3])

    # Define the branch pathway layers
    input_branch = Input(shape=(32, 32, 32))
    conv_branch = Conv2D(64, (1, 1))(input_branch)
    conv_branch = Conv2D(64, (3, 3), padding='same')(conv_branch)

    # Process the input through the branch pathway
    x_branch = Flatten()(input_branch)
    x_branch = Dense(128)(x_branch)
    x_branch = Dropout(0.2)(x_branch)

    # Combine the outputs from the main and branch pathways
    x_output = Add()([x_main, x_branch])
    x_output = Flatten()(x_output)
    output = Dense(10, activation='softmax')(x_output)

    # Define the model
    model = Model(inputs=[input_main, input_branch], outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model