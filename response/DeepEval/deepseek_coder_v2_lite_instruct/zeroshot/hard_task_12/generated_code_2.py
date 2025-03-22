import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Add, Flatten, Dense, Concatenate

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 64)

    # Define the inputs
    inputs = Input(shape=input_shape)

    # Main path
    # 1x1 convolutional layer for dimensionality reduction
    main_path = Conv2D(32, kernel_size=(1, 1), activation='relu')(inputs)

    # Two parallel convolutional layers
    conv1 = Conv2D(32, kernel_size=(1, 1), activation='relu')(main_path)
    conv2 = Conv2D(32, kernel_size=(3, 3), activation='relu')(main_path)

    # Concatenate the outputs of the two convolutional layers
    main_output = Concatenate()([conv1, conv2])

    # Branch path
    # 3x3 convolutional layer to match the channel dimension
    branch_path = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)

    # Add the main path and branch path outputs
    added_output = Add()([main_output, branch_path])

    # Flatten the output
    flattened = Flatten()(added_output)

    # Two fully connected layers
    fc1 = Dense(128, activation='relu')(flattened)
    fc2 = Dense(10, activation='softmax')(fc1)  # Assuming 10 classes for classification

    # Create the model
    model = Model(inputs=inputs, outputs=fc2)

    return model

# Example usage
model = dl_model()
model.summary()