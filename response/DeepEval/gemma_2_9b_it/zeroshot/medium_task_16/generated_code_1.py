import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Lambda, Flatten, Dense

def dl_model():
    model = Sequential()

    # Split the input channels into three groups
    model.add(Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3), input_shape=(32, 32, 3)))

    # Apply 1x1 convolutions to each group
    for i in range(3):
        model.add(Conv2D(filters=3, kernel_size=(1, 1), activation='relu', input_shape=(32, 32, 3)))

    # Downsample each group
    for i in range(3):
        model.add(AveragePooling2D(pool_size=(2, 2)))

    # Concatenate the feature maps
    model.add(Lambda(lambda x: tf.concat(x, axis=3)))

    # Flatten the feature maps
    model.add(Flatten())

    # Add two fully connected layers
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    return model