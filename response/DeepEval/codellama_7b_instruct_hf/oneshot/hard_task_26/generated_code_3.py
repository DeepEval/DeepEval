import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the model architecture
    main_path = keras.Sequential([
        # 1x1 convolutional layer for feature extraction
        Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'),

        # Branch 1: 3x3 convolutional layer for feature extraction
        Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),

        # Branch 2: Max pooling layer with 3x3 convolutional layer
        MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'),
        Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),

        # Branch 3: Max pooling layer with 3x3 convolutional layer
        MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'),
        Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),

        # Concatenate outputs from all branches
        Concatenate(),

        # 1x1 convolutional layer for final output
        Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'),

        # Batch normalization layer
        BatchNormalization(),

        # Flatten layer
        Flatten(),

        # Fully connected layers
        Dense(units=128, activation='relu'),
        Dense(units=10, activation='softmax')
    ])

    branch_path = keras.Sequential([
        # 1x1 convolutional layer for feature extraction
        Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'),

        # Max pooling layer
        MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'),

        # 3x3 convolutional layer for feature extraction
        Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),

        # Batch normalization layer
        BatchNormalization(),

        # Flatten layer
        Flatten(),

        # Fully connected layers
        Dense(units=128, activation='relu'),
        Dense(units=10, activation='softmax')
    ])

    # Add the branch path to the main path
    main_path.add(branch_path)

    # Define the input layer
    input_layer = Input(shape=input_shape)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=main_path(input_layer))

    return model