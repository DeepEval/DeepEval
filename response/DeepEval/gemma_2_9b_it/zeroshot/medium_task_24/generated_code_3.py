import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    input_tensor = layers.Input(shape=(32, 32, 3))

    # Branch 1
    branch1 = layers.Conv2D(32, (1, 1))(input_tensor)
    branch1 = layers.Conv2D(64, (3, 3), activation='relu')(branch1)
    branch1 = layers.Dropout(0.25)(branch1)

    # Branch 2
    branch2 = layers.Conv2D(32, (1, 1))(input_tensor)
    branch2 = layers.Conv2D(64, (1, 7), activation='relu')(branch2)
    branch2 = layers.Conv2D(64, (7, 1), activation='relu')(branch2)
    branch2 = layers.Conv2D(128, (3, 3), activation='relu')(branch2)
    branch2 = layers.Dropout(0.25)(branch2)

    # Branch 3
    branch3 = layers.MaxPooling2D((2, 2))(input_tensor)
    branch3 = layers.Conv2D(128, (3, 3), activation='relu')(branch3)
    branch3 = layers.Dropout(0.25)(branch3)

    # Concatenate branches
    merged = layers.concatenate([branch1, branch2, branch3], axis=-1)

    # Flatten and fully connected layers
    merged = layers.Flatten()(merged)
    merged = layers.Dense(512, activation='relu')(merged)
    merged = layers.Dropout(0.5)(merged)
    output = layers.Dense(10, activation='softmax')(merged)

    model = tf.keras.Model(inputs=input_tensor, outputs=output)

    return model