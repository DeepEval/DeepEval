import tensorflow as tf
from tensorflow import keras

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the lambda layer that splits the input into three groups
    split_layer = keras.layers.Lambda(lambda x: tf.split(x, 3, axis=1))

    # Define the first group of convolutional layers
    conv1_1 = keras.layers.Conv2D(32, (1, 1), activation='relu', input_shape=input_shape)
    conv1_2 = keras.layers.Conv2D(32, (3, 3), activation='relu')
    pool1 = keras.layers.MaxPooling2D((2, 2))

    # Define the second group of convolutional layers
    conv2_1 = keras.layers.Conv2D(64, (1, 1), activation='relu')
    conv2_2 = keras.layers.Conv2D(64, (3, 3), activation='relu')
    pool2 = keras.layers.MaxPooling2D((2, 2))

    # Define the third group of convolutional layers
    conv3_1 = keras.layers.Conv2D(128, (1, 1), activation='relu')
    conv3_2 = keras.layers.Conv2D(128, (3, 3), activation='relu')
    pool3 = keras.layers.MaxPooling2D((2, 2))

    # Define the branch pathway
    branch_conv1 = keras.layers.Conv2D(128, (1, 1), activation='relu')
    branch_conv2 = keras.layers.Conv2D(128, (3, 3), activation='relu')
    branch_pool = keras.layers.MaxPooling2D((2, 2))

    # Define the main pathway
    main_conv1 = keras.layers.Conv2D(128, (1, 1), activation='relu')
    main_conv2 = keras.layers.Conv2D(128, (3, 3), activation='relu')
    main_pool = keras.layers.MaxPooling2D((2, 2))

    # Define the dropout layer
    dropout = keras.layers.Dropout(0.5)

    # Define the fully connected layer
    flatten = keras.layers.Flatten()
    dense = keras.layers.Dense(10, activation='softmax')

    # Define the model
    model = keras.models.Sequential([
        split_layer,
        conv1_1, conv1_2, pool1,
        conv2_1, conv2_2, pool2,
        conv3_1, conv3_2, pool3,
        branch_conv1, branch_conv2, branch_pool,
        main_conv1, main_conv2, main_pool,
        dropout,
        flatten,
        dense
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model