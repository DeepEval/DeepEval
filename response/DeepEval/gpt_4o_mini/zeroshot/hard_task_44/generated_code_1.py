import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(32, 32, 3))

    # Block 1
    # Split the input into three groups along the channel axis
    split_inputs = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)

    # Feature extraction with different convolutional kernels
    conv1 = layers.Conv2D(32, (1, 1), activation='relu')(split_inputs[0])
    conv2 = layers.Conv2D(32, (3, 3), activation='relu')(split_inputs[1])
    conv3 = layers.Conv2D(32, (5, 5), activation='relu')(split_inputs[2])

    # Apply dropout to reduce overfitting
    drop1 = layers.Dropout(0.5)(conv1)
    drop2 = layers.Dropout(0.5)(conv2)
    drop3 = layers.Dropout(0.5)(conv3)

    # Concatenate the outputs from the three groups
    block1_output = layers.Concatenate()([drop1, drop2, drop3])

    # Block 2
    branch1 = layers.Conv2D(32, (1, 1), activation='relu')(block1_output)
    
    branch2 = layers.Conv2D(32, (1, 1), activation='relu')(block1_output)
    branch2 = layers.Conv2D(32, (3, 3), activation='relu')(branch2)

    branch3 = layers.Conv2D(32, (1, 1), activation='relu')(block1_output)
    branch3 = layers.Conv2D(32, (5, 5), activation='relu')(branch3)

    branch4 = layers.MaxPooling2D(pool_size=(3, 3), padding='same')(block1_output)
    branch4 = layers.Conv2D(32, (1, 1), activation='relu')(branch4)

    # Concatenate the outputs from all branches for feature fusion
    block2_output = layers.Concatenate()([branch1, branch2, branch3, branch4])

    # Flatten the output
    flatten = layers.Flatten()(block2_output)

    # Fully connected layer
    outputs = layers.Dense(10, activation='softmax')(flatten)

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model