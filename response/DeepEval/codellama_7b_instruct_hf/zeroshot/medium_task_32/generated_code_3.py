import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize the input data
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the model
    model = keras.Sequential()
    model.add(layers.Lambda(lambda x: tf.split(x, 3, axis=-1),
                            input_shape=input_shape))
    model.add(layers.DepthwiseSeparableConv2D(1, 1, activation='relu'))
    model.add(layers.DepthwiseSeparableConv2D(3, 3, activation='relu'))
    model.add(layers.DepthwiseSeparableConv2D(5, 5, activation='relu'))
    model.add(layers.Concatenate())
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

    return model