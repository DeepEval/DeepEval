import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Lambda, Concatenate, Flatten, Dense

def dl_model():
    # Load CIFAR-10 dataset (for demonstration purposes, we'll use placeholder inputs)
    inputs = Input(shape=(32, 32, 3))

    # Main path
    main_path = Conv2D(32, (1, 1), padding='same')(inputs)
    main_path_1x1 = Conv2D(32, (1, 1), padding='same', activation='relu')(main_path)
    main_path_3x3 = Conv2D(32, (3, 3), padding='same', activation='relu')(main_path)
    main_path_5x5 = Conv2D(32, (5, 5), padding='same', activation='relu')(main_path)

    # Split the main path into three groups
    split_1x1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1)[0])(main_path_1x1)
    split_3x3 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1)[1])(main_path_3x3)
    split_5x5 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1)[2])(main_path_5x5)

    # Concatenate the outputs from the three groups
    main_output = Concatenate(axis=-1)([split_1x1, split_3x3, split_5x5])

    # Branch path
    branch_path = Conv2D(32, (1, 1), padding='same')(inputs)

    # Add the main and branch paths
    added = Add()([main_output, branch_path])

    # Flatten the output
    flattened = Flatten()(added)

    # Fully connected layers
    fc1 = Dense(128, activation='relu')(flattened)
    fc2 = Dense(10, activation='softmax')(fc1)

    # Define the model
    model = Model(inputs=inputs, outputs=fc2)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Example usage
if __name__ == "__main__":
    model = dl_model()
    model.summary()