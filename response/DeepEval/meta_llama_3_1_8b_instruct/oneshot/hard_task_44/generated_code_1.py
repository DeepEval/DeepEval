import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input layer
    input_layer = keras.Input(shape=(32, 32, 3))

    # Define block 1
    def block1(input_tensor):
        # Split the input into three groups along the channel
        group1 = layers.Lambda(lambda x: x[:, :, :, 0:1])(input_tensor)
        group2 = layers.Lambda(lambda x: x[:, :, :, 1:2])(input_tensor)
        group3 = layers.Lambda(lambda x: x[:, :, :, 2:3])(input_tensor)

        # Feature extraction through convolutional with varying kernel sizes
        conv1 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(group1)
        conv2 = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(group2)
        conv3 = layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(group3)

        # Concatenate the outputs from the three groups
        output_tensor = layers.Concatenate()([conv1, conv2, conv3])

        # Apply dropout to reduce overfitting
        output_tensor = layers.Dropout(0.2)(output_tensor)

        return output_tensor

    # Define block 1 output
    block1_output = block1(input_layer)

    # Define block 2
    def block2(input_tensor):
        # Define the four branches
        branch1 = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_tensor)
        branch2 = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_tensor)
        branch3 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(input_tensor)
        branch4 = layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(input_tensor)

        # Apply max pooling to branch 3
        branch3 = layers.MaxPooling2D(pool_size=(2, 2))(branch3)
        branch3 = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(branch3)

        # Concatenate the outputs from all branches
        output_tensor = layers.Concatenate()([branch1, branch2, branch3, branch4])

        return output_tensor

    # Define block 2 output
    block2_output = block2(block1_output)

    # Apply batch normalization and flatten the result
    batch_norm = layers.BatchNormalization()(block2_output)
    flatten_layer = layers.Flatten()(batch_norm)

    # Define the output layer
    output_layer = layers.Dense(units=10, activation='softmax')(flatten_layer)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model