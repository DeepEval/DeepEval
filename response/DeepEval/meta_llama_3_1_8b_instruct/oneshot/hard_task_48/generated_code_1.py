import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape of the CIFAR-10 dataset
    input_shape = (32, 32, 3)

    # Define the input layer
    input_layer = keras.Input(shape=input_shape)

    # Define Block 1
    def block1(input_tensor):
        # Split the input into three groups using tf.split within Lambda layer
        split_input = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        # Apply separable convolutional layers with different kernel sizes (1x1, 3x3, 5x5)
        conv1 = layers.SeparableConv2D(filters=64, kernel_size=(1, 1), activation='relu')(split_input[0])
        conv2 = layers.SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu')(split_input[1])
        conv3 = layers.SeparableConv2D(filters=64, kernel_size=(5, 5), activation='relu')(split_input[2])
        # Concatenate the outputs of the three groups
        output_tensor = layers.Concatenate()([conv1, conv2, conv3])
        # Apply batch normalization to enhance model performance
        output_tensor = layers.BatchNormalization()(output_tensor)
        return output_tensor

    # Apply Block 1
    block1_output = block1(input_layer)
    # Apply Block 2
    def block2(input_tensor):
        # Define Path 1
        path1 = layers.SeparableConv2D(filters=128, kernel_size=(1, 1), activation='relu')(input_tensor)
        # Define Path 2
        path2 = layers.AveragePooling2D(pool_size=(3, 3), strides=2, padding='same')(input_tensor)
        path2 = layers.SeparableConv2D(filters=128, kernel_size=(1, 1), activation='relu')(path2)
        # Define Path 3
        path3 = layers.SeparableConv2D(filters=128, kernel_size=(1, 1), activation='relu')(input_tensor)
        path3 = layers.Concatenate()([layers.Conv2D(filters=128, kernel_size=(1, 3), activation='relu')(path3), 
                                      layers.Conv2D(filters=128, kernel_size=(3, 1), activation='relu')(path3)])
        # Define Path 4
        path4 = layers.SeparableConv2D(filters=128, kernel_size=(1, 1), activation='relu')(input_tensor)
        path4 = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(path4)
        path4 = layers.Concatenate()([layers.Conv2D(filters=128, kernel_size=(1, 3), activation='relu')(path4), 
                                      layers.Conv2D(filters=128, kernel_size=(3, 1), activation='relu')(path4)])
        # Concatenate the outputs of the four paths
        output_tensor = layers.Concatenate()([path1, path2, path3, path4])
        return output_tensor

    # Apply Block 2
    block2_output = block2(block1_output)
    # Apply batch normalization to enhance model performance
    batch_norm = layers.BatchNormalization()(block2_output)
    # Apply a flattening layer to flatten the feature map
    flatten_layer = layers.Flatten()(batch_norm)
    # Apply a fully connected layer to output the classification result
    output_layer = layers.Dense(10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model