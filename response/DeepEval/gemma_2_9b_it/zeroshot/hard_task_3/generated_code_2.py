import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    input_tensor = layers.Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    split_tensor = layers.Lambda(lambda x: tf.split(x, 3, axis=2))(input_tensor)

    # Create branches for each group
    branch1 = tf.keras.Sequential([
        layers.Conv2D(32, (1, 1), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Dropout(0.5)
    ])(split_tensor[0])

    branch2 = tf.keras.Sequential([
        layers.Conv2D(32, (1, 1), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Dropout(0.5)
    ])(split_tensor[1])

    branch3 = tf.keras.Sequential([
        layers.Conv2D(32, (1, 1), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Dropout(0.5)
    ])(split_tensor[2])

    # Concatenate outputs from the three branches
    main_pathway = layers.concatenate([branch1, branch2, branch3], axis=3)

    # Parallel branch
    branch_parallel = tf.keras.Sequential([
        layers.Conv2D(64, (1, 1), activation='relu')
    ])(input_tensor)

    # Combine outputs from main pathway and parallel branch
    combined_output = layers.add([main_pathway, branch_parallel])

    # Flatten and classify
    output = layers.Flatten()(combined_output)
    output = layers.Dense(10, activation='softmax')(output)

    model = tf.keras.Model(inputs=input_tensor, outputs=output)
    
    return model