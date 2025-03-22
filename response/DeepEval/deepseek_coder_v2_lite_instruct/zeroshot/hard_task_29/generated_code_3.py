import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Concatenate

def dl_model():
    # Define the input layer
    inputs = Input(shape=(28, 28, 1))

    # First block
    # Main path
    main_path = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    main_path = Conv2D(32, (3, 3), activation='relu')(main_path)

    # Branch path
    branch_path = Conv2D(32, (1, 1), activation='relu')(inputs)

    # Combine outputs from both paths
    combined_output = Add()([main_path, branch_path])

    # Second block
    # Pooling layers
    pool1 = MaxPooling2D((1, 1))(combined_output)
    pool2 = MaxPooling2D((2, 2))(combined_output)
    pool3 = MaxPooling2D((4, 4))(combined_output)

    # Flatten the results
    flattened1 = Flatten()(pool1)
    flattened2 = Flatten()(pool2)
    flattened3 = Flatten()(pool3)

    # Concatenate the flattened results
    concatenated_output = Concatenate()([flattened1, flattened2, flattened3])

    # Fully connected layers
    fc1 = Dense(128, activation='relu')(concatenated_output)
    fc2 = Dense(10, activation='softmax')(fc1)

    # Define the model
    model = Model(inputs=inputs, outputs=fc2)

    return model

# Example usage:
# model = dl_model()
# model.summary()