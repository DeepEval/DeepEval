import tensorflow as tf
from tensorflow.keras import layers, models, Input

def dl_model():
    # Input layer for CIFAR-10 images (32x32x3)
    inputs = Input(shape=(32, 32, 3))
    
    # Block 1: Splitting into three groups with separable convolutions
    split_inputs = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    
    # Group 1: 1x1 Separable Convolution
    group1 = layers.SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same')(split_inputs[0])
    group1 = layers.BatchNormalization()(group1)
    group1 = layers.Activation('relu')(group1)
    
    # Group 2: 3x3 Separable Convolution
    group2 = layers.SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same')(split_inputs[1])
    group2 = layers.BatchNormalization()(group2)
    group2 = layers.Activation('relu')(group2)
    
    # Group 3: 5x5 Separable Convolution
    group3 = layers.SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same')(split_inputs[2])
    group3 = layers.BatchNormalization()(group3)
    group3 = layers.Activation('relu')(group3)
    
    # Concatenate outputs of Block 1
    block1_output = layers.concatenate([group1, group2, group3])
    
    # Block 2: Four parallel branches
    # Path 1
    path1 = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')(block1_output)
    
    # Path 2
    path2 = layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(block1_output)
    path2 = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')(path2)

    # Path 3
    path3 = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')(block1_output)
    path3_1 = layers.Conv2D(filters=32, kernel_size=(1, 3), padding='same')(path3)
    path3_2 = layers.Conv2D(filters=32, kernel_size=(3, 1), padding='same')(path3)
    path3 = layers.concatenate([path3_1, path3_2])

    # Path 4
    path4 = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')(block1_output)
    path4 = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')(path4)
    path4_1 = layers.Conv2D(filters=32, kernel_size=(1, 3), padding='same')(path4)
    path4_2 = layers.Conv2D(filters=32, kernel_size=(3, 1), padding='same')(path4)
    path4 = layers.concatenate([path4_1, path4_2])
    
    # Concatenate outputs of Block 2
    block2_output = layers.concatenate([path1, path2, path3, path4])
    
    # Final Layers
    flatten_output = layers.Flatten()(block2_output)
    output = layers.Dense(units=10, activation='softmax')(flatten_output)  # CIFAR-10 has 10 classes
    
    # Create model
    model = models.Model(inputs=inputs, outputs=output)
    
    return model