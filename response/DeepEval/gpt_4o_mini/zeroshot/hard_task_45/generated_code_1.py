import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model(input_shape=(32, 32, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)

    # First block: Splitting input into three groups and applying depthwise separable convolutions
    split_inputs = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)

    conv_1x1 = layers.SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_inputs[0])
    conv_3x3 = layers.SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_inputs[1])
    conv_5x5 = layers.SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_inputs[2])

    # Concatenating outputs from the first block
    concatenated_block_1 = layers.Concatenate()([conv_1x1, conv_3x3, conv_5x5])

    # Second block: Multiple branches for feature extraction
    branch_1 = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(concatenated_block_1)
    branch_2 = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(concatenated_block_1)
    branch_3 = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(concatenated_block_1)
    branch_4 = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch_3)
    
    # Branch with max pooling followed by 1x1 convolution
    max_pool = layers.MaxPooling2D(pool_size=(2, 2))(concatenated_block_1)
    branch_5 = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(max_pool)

    # Concatenating outputs from all branches
    concatenated_block_2 = layers.Concatenate()([branch_1, branch_2, branch_3, branch_4, branch_5])

    # Flattening and fully connected layer
    flatten = layers.Flatten()(concatenated_block_2)
    outputs = layers.Dense(num_classes, activation='softmax')(flatten)

    # Constructing the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
model = dl_model()
model.summary()  # To visualize the model architecture