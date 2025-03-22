import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Concatenate, Reshape

def dl_model():
    # Input layer
    inputs = Input(shape=(28, 28, 1))

    # Block 1
    # Path 1: 1x1 Average Pooling
    path1 = AveragePooling2D(pool_size=(1, 1), strides=1)(inputs)
    path1 = Flatten()(path1)

    # Path 2: 2x2 Average Pooling
    path2 = AveragePooling2D(pool_size=(2, 2), strides=2)(inputs)
    path2 = Flatten()(path2)

    # Path 3: 4x4 Average Pooling
    path3 = AveragePooling2D(pool_size=(4, 4), strides=4)(inputs)
    path3 = Flatten()(path3)

    # Concatenate outputs of the three paths
    combined_output = Concatenate(axis=-1)([path1, path2, path3])

    # Fully connected layer after Block 1
    fc_layer = Dense(128, activation='relu')(combined_output)

    # Reshape output to 4-dimensional tensor
    reshaped_output = Reshape((1, 1, 128))(fc_layer)

    # Block 2
    # Branch 1: 1x1 Conv -> 3x3 Conv
    branch1 = Conv2D(32, (1, 1), activation='relu')(reshaped_output)
    branch1 = Conv2D(32, (3, 3), activation='relu')(branch1)

    # Branch 2: 1x1 Conv -> 1x7 Conv -> 7x1 Conv
    branch2 = Conv2D(32, (1, 1), activation='relu')(reshaped_output)
    branch2 = Conv2D(32, (1, 7), activation='relu')(branch2)
    branch2 = Conv2D(32, (7, 1), activation='relu')(branch2)

    # Branch 3: 3x3 Conv -> Average Pooling
    branch3 = Conv2D(32, (3, 3), activation='relu')(reshaped_output)
    branch3 = AveragePooling2D(pool_size=(3, 3), strides=3)(branch3)

    # Concatenate outputs of the three branches
    concatenated_features = Concatenate(axis=-1)([branch1, branch2, branch3])

    # Flatten the concatenated features
    flattened_features = Flatten()(concatenated_features)

    # Fully connected layers
    fc1 = Dense(64, activation='relu')(flattened_features)
    outputs = Dense(10, activation='softmax')(fc1)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
model = dl_model()
model.summary()