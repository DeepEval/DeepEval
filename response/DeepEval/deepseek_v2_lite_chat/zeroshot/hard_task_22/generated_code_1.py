import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Flatten, Dense


def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Splitting the input into three groups along the channel
    x_train_split1, x_train_split2, x_train_split3 = tf.split(x_train, 3, axis=-1)
    x_test_split1, x_test_split2, x_test_split3 = tf.split(x_test, 3, axis=-1)

    # Main path of the model
    input_main_path = tf.keras.Input(shape=(32, 32, 3))

    # Separable convolutional layers
    x = Conv2D(32, (1, 1), activation='relu')(input_main_path)
    x = SeparableConv2D(32, (3, 3), activation='relu')(x)
    x = SeparableConv2D(32, (5, 5), activation='relu')(x)

    # Branch path
    x_branch = Conv2D(32, (1, 1), activation='relu')(input_main_path)

    # Fusing the outputs from the main and branch paths
    x = Concatenate()([x, x_branch])

    # Fully connected layers for the classification task
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(10, activation='softmax')(x)

    # Constructing the model
    model = Model(inputs=[input_main_path, input_branch_path], outputs=output)

    # Compiling the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Return the constructed model
    return model