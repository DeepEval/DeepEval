import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_tensor = layers.Input(shape=(32, 32, 3))

    # First block
    # Split the input into three groups along the channel axis
    split_inputs = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)

    # Apply separable convolutions with different kernel sizes
    conv1 = layers.SeparableConv2D(filters=32, kernel_size=(1, 1), activation='relu')(split_inputs[0])
    conv2 = layers.SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu')(split_inputs[1])
    conv3 = layers.SeparableConv2D(filters=32, kernel_size=(5, 5), activation='relu')(split_inputs[2])

    # Concatenate outputs from the three separable convolutions
    concatenated_block1 = layers.Concatenate(axis=-1)([conv1, conv2, conv3])

    # Second block
    # Branch 1: 3x3 Convolution
    branch1 = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(concatenated_block1)

    # Branch 2: 1x1 Convolution followed by two 3x3 Convolutions
    branch2 = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(concatenated_block1)
    branch2 = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
    branch2 = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch2)

    # Branch 3: Max Pooling
    branch3 = layers.MaxPooling2D(pool_size=(2, 2))(concatenated_block1)

    # Concatenate outputs from all branches
    concatenated_block2 = layers.Concatenate(axis=-1)([branch1, branch2, branch3])

    # Global Average Pooling
    global_avg_pool = layers.GlobalAveragePooling2D()(concatenated_block2)

    # Fully connected layer for classification
    output = layers.Dense(10, activation='softmax')(global_avg_pool)

    # Create the model
    model = models.Model(inputs=input_tensor, outputs=output)

    return model

# Example of creating the model
model = dl_model()
model.summary()  # This will display the model architecture