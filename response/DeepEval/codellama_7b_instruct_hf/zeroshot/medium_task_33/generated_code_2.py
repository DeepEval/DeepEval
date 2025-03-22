import keras
from keras.layers import Input, Lambda, Concatenate, Dense
from keras.applications.vgg16 import VGG16

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the model
    model = keras.Sequential()

    # Add the first group of layers
    model.add(Lambda(lambda x: tf.split(x, 3, axis=2), input_shape=input_shape))
    model.add(Concatenate())
    model.add(SeparableConv2D(64, 1, activation='relu'))
    model.add(SeparableConv2D(128, 3, activation='relu'))
    model.add(SeparableConv2D(256, 5, activation='relu'))
    model.add(Concatenate())

    # Add the second group of layers
    model.add(Lambda(lambda x: tf.split(x, 3, axis=2), input_shape=input_shape))
    model.add(Concatenate())
    model.add(SeparableConv2D(64, 1, activation='relu'))
    model.add(SeparableConv2D(128, 3, activation='relu'))
    model.add(SeparableConv2D(256, 5, activation='relu'))
    model.add(Concatenate())

    # Add the third group of layers
    model.add(Lambda(lambda x: tf.split(x, 3, axis=2), input_shape=input_shape))
    model.add(Concatenate())
    model.add(SeparableConv2D(64, 1, activation='relu'))
    model.add(SeparableConv2D(128, 3, activation='relu'))
    model.add(SeparableConv2D(256, 5, activation='relu'))
    model.add(Concatenate())

    # Add the final fully connected layers
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model