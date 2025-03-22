import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

def dl_model():
    # Input layer
    input_img = layers.Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    split_input = layers.Lambda(tf.split, arguments={'axis': 3, 'num_or_size_splits': 3})(input_img)

    # Define the main path
    main_path = layers.Conv2D(32, (1, 1), activation='relu')(split_input[0])
    main_path = layers.Conv2D(64, (3, 3), activation='relu')(main_path)
    main_path = layers.Conv2D(128, (1, 1), activation='relu')(main_path)

    # Define the side path
    side_path = layers.Conv2D(16, (1, 1), activation='relu')(split_input[1])
    side_path = layers.Conv2D(32, (3, 3), activation='relu')(side_path)

    # Combine the main and side paths
    output = layers.add([main_path, side_path])

    # Fuse the output with the original input
    output = layers.add([output, input_img])

    # Flatten the output
    output = layers.Flatten()(output)

    # Fully connected layer for classification
    output = layers.Dense(10, activation='softmax')(output)

    # Create the model
    model = models.Model(inputs=input_img, outputs=output)

    return model

# Create the model
model = dl_model()