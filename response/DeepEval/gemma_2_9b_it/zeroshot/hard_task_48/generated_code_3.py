import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    input_tensor = layers.Input(shape=(32, 32, 3))

    # Block 1
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=1))(input_tensor)  
    
    # Apply separable convolutions and batch normalization to each split
    branch1 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(x[0])
    branch1 = layers.BatchNormalization()(branch1)
    branch2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x[1])
    branch2 = layers.BatchNormalization()(branch2)
    branch3 = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(x[2])
    branch3 = layers.BatchNormalization()(branch3)

    # Concatenate outputs from different kernel sizes
    x = layers.concatenate([branch1, branch2, branch3], axis=1)

    # Block 2
    # Path 1
    path1 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(x)
    path1 = layers.BatchNormalization()(path1)

    # Path 2
    path2 = layers.AveragePooling2D((3, 3))(x)
    path2 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(path2)
    path2 = layers.BatchNormalization()(path2)

    # Path 3
    path3 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(x)
    path3 = layers.BatchNormalization()(path3)
    path3_1 = layers.Conv2D(32, (1, 3), activation='relu', padding='same')(path3)
    path3_1 = layers.BatchNormalization()(path3_1)
    path3_2 = layers.Conv2D(32, (3, 1), activation='relu', padding='same')(path3)
    path3_2 = layers.BatchNormalization()(path3_2)
    path3 = layers.concatenate([path3_1, path3_2], axis=1)

    # Path 4
    path4 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(x)
    path4 = layers.BatchNormalization()(path4)
    path4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(path4)
    path4 = layers.BatchNormalization()(path4)
    path4_1 = layers.Conv2D(32, (1, 3), activation='relu', padding='same')(path4)
    path4_1 = layers.BatchNormalization()(path4_1)
    path4_2 = layers.Conv2D(32, (3, 1), activation='relu', padding='same')(path4)
    path4_2 = layers.BatchNormalization()(path4_2)
    path4 = layers.concatenate([path4_1, path4_2], axis=1)

    # Concatenate outputs from all paths
    x = layers.concatenate([path1, path2, path3, path4], axis=1)

    # Classification layer
    x = layers.Flatten()(x)
    x = layers.Dense(10, activation='softmax')(x)  

    model = models.Model(inputs=input_tensor, outputs=x)

    return model