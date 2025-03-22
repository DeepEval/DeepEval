import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    model = tf.keras.models.Sequential()

    # Input Layer
    input_tensor = layers.Input(shape=(32, 32, 3))  

    # First Block - Split and Depthwise Separable Convolutions
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)
    
    # Group 1
    x1 = layers.Conv2D(32, (1, 1), activation='relu', padding='same', name='conv1_1')(x[0])
    x1 = layers.BatchNormalization()(x1)

    # Group 2
    x2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2_1')(x[1])
    x2 = layers.BatchNormalization()(x2)

    # Group 3
    x3 = layers.Conv2D(32, (5, 5), activation='relu', padding='same', name='conv3_1')(x[2])
    x3 = layers.BatchNormalization()(x3)

    x = layers.concatenate([x1, x2, x3], axis=-1)

    # Second Block - Multiple Branches and Concatenation
    # Branch 1
    branch1 = layers.Conv2D(64, (1, 1), activation='relu', padding='same', name='branch1_1')(x)
    branch1 = layers.BatchNormalization()(branch1)

    # Branch 2
    branch2 = layers.Conv2D(64, (1, 1), activation='relu', padding='same', name='branch2_1')(x)
    branch2 = layers.BatchNormalization()(branch2)
    branch2 = layers.Conv2D(64, (1, 7), activation='relu', padding='same', name='branch2_2')(branch2)
    branch2 = layers.BatchNormalization()(branch2)
    branch2 = layers.Conv2D(64, (7, 1), activation='relu', padding='same', name='branch2_3')(branch2)
    branch2 = layers.BatchNormalization()(branch2)
    branch2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='branch2_4')(branch2)
    branch2 = layers.BatchNormalization()(branch2)

    # Branch 3
    branch3 = layers.AveragePooling2D((2, 2), strides=2, padding='same')(x)
    branch3 = layers.Conv2D(64, (1, 1), activation='relu', padding='same', name='branch3_1')(branch3)
    branch3 = layers.BatchNormalization()(branch3)

    # Concatenate Branches
    x = layers.concatenate([branch1, branch2, branch3], axis=-1)

    # Fully Connected Layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x) 

    model = tf.keras.models.Model(inputs=input_tensor, outputs=outputs)

    return model