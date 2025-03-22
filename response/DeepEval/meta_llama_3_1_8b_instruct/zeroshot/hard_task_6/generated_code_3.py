import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape of the CIFAR-10 dataset
    input_shape = (32, 32, 3)

    # Create the input layer
    inputs = keras.Input(shape=input_shape)

    # Define Block 1
    def block1(x):
        x = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(x)
        group1 = layers.Conv2D(64 // 3, (1, 1), activation='relu')(x[0])
        group2 = layers.Conv2D(64 // 3, (1, 1), activation='relu')(x[1])
        group3 = layers.Conv2D(64 // 3, (1, 1), activation='relu')(x[2])
        return layers.Concatenate()([group1, group2, group3])

    # Apply Block 1
    x = block1(inputs)

    # Define Block 2
    def block2(x):
        _, height, width, channels = x.shape.as_list()
        x = layers.Reshape((height, width, 3, channels // 3))(x)
        x = layers.Permutation((3, 4))(x)
        return layers.Reshape(x.shape.as_list()[:3] + [x.shape.as_list()[4]])(x)

    # Apply Block 2
    x = block2(x)

    # Define Block 3
    def block3(x):
        return layers.SeparableConv2D(128, (3, 3), activation='relu')(x)

    # Apply Block 3
    x = block3(x)

    # Repeat Block 1
    x = block1(x)

    # Extract features from the original input using average pooling
    x_avg = layers.AveragePooling2D((2, 2))(inputs)

    # Concatenate the outputs from both paths
    x = layers.Concatenate()([x, x_avg])

    # Apply a fully connected layer to complete the classification process
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# Call the function to construct the model
model = dl_model()
print(model.summary())