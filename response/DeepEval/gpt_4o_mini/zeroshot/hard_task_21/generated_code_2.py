import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel axis
    split_inputs = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Main path: Each split goes through a depthwise separable convolution
    def create_branch(input_tensor, kernel_size):
        x = layers.SeparableConv2D(filters=32, kernel_size=kernel_size, padding='same', activation='relu')(input_tensor)
        x = layers.BatchNormalization()(x)
        return x

    # Creating branches with different kernel sizes
    branch_1 = create_branch(split_inputs[0], (1, 1))
    branch_2 = create_branch(split_inputs[1], (3, 3))
    branch_3 = create_branch(split_inputs[2], (5, 5))

    # Concatenate the outputs from the three branches
    main_path_output = layers.Concatenate()([branch_1, branch_2, branch_3])

    # Branch path: 1x1 convolution to align channels
    branch_path_output = layers.Conv2D(filters=tf.shape(main_path_output)[-1], kernel_size=(1, 1), padding='same', activation='relu')(main_path_output)

    # Combine main path and branch path with addition
    combined_output = layers.Add()([main_path_output, branch_path_output])

    # Flatten the output for the fully connected layers
    flatten_output = layers.Flatten()(combined_output)

    # Fully connected layers
    dense_1 = layers.Dense(128, activation='relu')(flatten_output)
    dense_2 = layers.Dense(10, activation='softmax')(dense_1)  # 10 classes for CIFAR-10

    # Creating the model
    model = models.Model(inputs=input_layer, outputs=dense_2)

    return model