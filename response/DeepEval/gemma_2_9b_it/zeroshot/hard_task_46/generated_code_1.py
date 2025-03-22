import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    input_tensor = layers.Input(shape=(32, 32, 3))

    # First Block
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=2))(input_tensor)
    
    # Group 1
    group1 = layers.Conv2D(64, (1, 1), activation='relu')(x[0])
    
    # Group 2
    group2 = layers.SeparableConv2D(64, (3, 3), activation='relu')(x[1])
    
    # Group 3
    group3 = layers.SeparableConv2D(64, (5, 5), activation='relu')(x[2])

    x = tf.concat([group1, group2, group3], axis=2)

    # Second Block
    branch1 = layers.Conv2D(128, (3, 3), activation='relu')(x)
    branch2 = layers.Sequential([
        layers.Conv2D(128, (1, 1), activation='relu'),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Conv2D(128, (3, 3), activation='relu')
    ])(x)
    branch3 = layers.MaxPooling2D((2, 2))(x)

    x = tf.concat([branch1, branch2, branch3], axis=3)

    # Global Average Pooling and Output Layer
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(10, activation='softmax')(x)

    model = models.Model(inputs=input_tensor, outputs=x)

    return model