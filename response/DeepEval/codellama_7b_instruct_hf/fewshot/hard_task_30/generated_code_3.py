import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

 和 return model
def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the dual-path structure for the first block
    main_path = layers.Conv2D(32, (3, 3), activation='relu')(input_shape)
    main_path = layers.Conv2D(64, (3, 3), activation='relu')(main_path)
    main_path = layers.MaxPooling2D((2, 2))(main_path)

    branch_path = layers.Conv2D(32, (3, 3), activation='relu')(input_shape)
    branch_path = layers.MaxPooling2D((2, 2))(branch_path)

    # Add the main and branch paths
    output_path = layers.Add()([main_path, branch_path])

    # Define the second block
    input_shape = (32, 32, 64)

    # Split the input into three groups
    split_input = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(input_shape)

    # Define the depthwise separable convolutional layers for each group
    group1 = layers.DepthwiseSeparableConv2D(1, (1, 1), activation='relu')(split_input[0])
    group2 = layers.DepthwiseSeparableConv2D(3, (3, 3), activation='relu')(split_input[1])
    group3 = layers.DepthwiseSeparableConv2D(5, (5, 5), activation='relu')(split_input[2])

    # Concatenate the output of the groups
    output_path = layers.Concatenate()([group1, group2, group3])

    # Define the fully connected layers for classification
    output_path = layers.Dense(64, activation='relu')(output_path)
    output_path = layers.Dense(10, activation='softmax')(output_path)

    # Create the model
    model = keras.Model(input_shape, output_path)

    return model