import tensorflow as tf
from tensorflow import keras

def dl_model():
    # Create the input layer
    input_layer = keras.Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel
    main_path = input_layer
    branch_path = keras.layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)

    # Apply multi-scale feature extraction to each group
    main_path = keras.layers.Conv2D(32, (1, 1), activation='relu')(main_path)
    main_path = keras.layers.Conv2D(32, (3, 3), activation='relu')(main_path)
    main_path = keras.layers.Conv2D(32, (5, 5), activation='relu')(main_path)
    main_path = keras.layers.MaxPooling2D((2, 2))(main_path)
    main_path = keras.layers.Flatten()(main_path)

    branch_path = keras.layers.Conv2D(32, (1, 1), activation='relu')(branch_path)
    branch_path = keras.layers.MaxPooling2D((2, 2))(branch_path)
    branch_path = keras.layers.Flatten()(branch_path)

    # Concat the outputs from both paths
    concatenated = keras.layers.concatenate([main_path, branch_path])

    # Apply two fully connected layers for classification
    concatenated = keras.layers.Dense(128, activation='relu')(concatenated)
    concatenated = keras.layers.Dense(10, activation='softmax')(concatenated)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=concatenated)

    # Compile the model with a loss function and optimizer
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model