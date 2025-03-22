import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    
    input_tensor = layers.Input(shape=(32, 32, 3))  # Input shape for CIFAR-10

    # Branch 1: 1x1 convolution for dimensionality reduction
    branch1 = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_tensor)

    # Branch 2: 1x1 convolution followed by 3x3 convolution
    branch2 = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_tensor)
    branch2 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch2)

    # Branch 3: 1x1 convolution followed by 5x5 convolution
    branch3 = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_tensor)
    branch3 = layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(branch3)

    # Branch 4: 3x3 max pooling followed by 1x1 convolution
    branch4 = layers.MaxPooling2D(pool_size=(3, 3))(input_tensor)
    branch4 = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(branch4)

    # Concatenate the outputs of all branches
    merged = layers.Concatenate()([branch1, branch2, branch3, branch4])

    # Flatten the concatenated features
    flattened = layers.Flatten()(merged)

    # Fully connected layers
    dense1 = layers.Dense(units=128, activation='relu')(flattened)
    output = layers.Dense(units=10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    model = tf.keras.Model(inputs=input_tensor, outputs=output)
    return model