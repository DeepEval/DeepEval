import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    inputs = layers.Input(shape=(32, 32, 3))

    # Main Path
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=1))(inputs) 
    
    # 1x1 Convolutional Branch
    branch_x = layers.Conv2D(filters=128, kernel_size=(1, 1))(inputs)

    # Feature Extraction in Main Path
    conv1 = layers.Conv2D(filters=64, kernel_size=(1, 1))(x[0])
    conv2 = layers.Conv2D(filters=64, kernel_size=(3, 3))(x[1])
    conv3 = layers.Conv2D(filters=64, kernel_size=(5, 5))(x[2])

    # Concatenate Outputs from Main Path
    x = layers.concatenate([conv1, conv2, conv3], axis=1) 

    # Fuse Main and Branch Paths
    x = layers.Add()([x, branch_x])

    # Classification Layers
    x = layers.Flatten()(x)
    x = layers.Dense(units=128, activation='relu')(x)
    outputs = layers.Dense(units=10, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model