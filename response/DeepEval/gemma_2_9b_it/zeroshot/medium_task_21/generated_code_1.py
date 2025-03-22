import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    input_tensor = layers.Input(shape=(32, 32, 3))

    # Branch 1: 1x1 convolution
    branch1 = layers.Conv2D(32, (1, 1), activation='relu')(input_tensor)
    branch1 = layers.Dropout(0.2)(branch1)

    # Branch 2: 1x1 convolution followed by 3x3 convolution
    branch2 = layers.Conv2D(32, (1, 1), activation='relu')(input_tensor)
    branch2 = layers.Conv2D(64, (3, 3), activation='relu')(branch2)
    branch2 = layers.Dropout(0.2)(branch2)

    # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
    branch3 = layers.Conv2D(32, (1, 1), activation='relu')(input_tensor)
    branch3 = layers.Conv2D(64, (3, 3), activation='relu')(branch3)
    branch3 = layers.Conv2D(128, (3, 3), activation='relu')(branch3)
    branch3 = layers.Dropout(0.2)(branch3)

    # Branch 4: Average pooling followed by 1x1 convolution
    branch4 = layers.AveragePooling2D((2, 2))(input_tensor)
    branch4 = layers.Conv2D(128, (1, 1), activation='relu')(branch4)
    branch4 = layers.Dropout(0.2)(branch4)

    # Concatenate the outputs from all branches
    merged = layers.concatenate([branch1, branch2, branch3, branch4], axis=-1)

    # Fully connected layers
    merged = layers.Flatten()(merged)
    merged = layers.Dense(128, activation='relu')(merged)
    merged = layers.Dropout(0.5)(merged)
    output = layers.Dense(10, activation='softmax')(merged)

    model = tf.keras.Model(inputs=input_tensor, outputs=output)

    return model