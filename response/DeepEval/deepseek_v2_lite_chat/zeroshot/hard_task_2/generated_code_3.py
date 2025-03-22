import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.layers import Lambda, Concatenate

def dl_model():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 32, 32, 3).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], 32, 32, 3).astype('float32') / 255

    # Define the Lambda layer to split the input into three groups
    split_layer = Lambda(lambda x: tf.split(x, 3, axis=-1))

    # Define the first 1x1 convolution layer
    conv1 = Conv2D(32, (1, 1), activation='relu', padding='same')(x_train[0])

    # Define the second 3x3 convolution layer
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)

    # Define the third 1x1 convolution layer
    conv3 = Conv2D(64, (1, 1), activation='relu', padding='same')(conv2)

    # Combine the outputs from the three groups using concatenation
    concat_layer = Concatenate(axis=-1)([tf.expand_dims(group, axis=1) for group in split_layer])

    # Add the main path to the original input
    main_path = Conv2D(64, (1, 1), activation='relu', padding='same')(tf.concat([x_train[0], conv1, conv2], axis=-1))

    # Combine the main path with the original input
    combined = Conv2D(64, (1, 1), activation='relu', padding='same')(tf.concat([main_path, x_train[0]], axis=-1))

    # Flatten the combined features
    flatten = Flatten()(combined)

    # Feed into a fully connected layer for classification
    output_layer = Dense(10, activation='softmax')(flatten)

    # Define the model
    model = Model(inputs=[x_train[0], x_train[1:]], outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Create the model
model = dl_model()