import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    input_layer = layers.Input(shape=(28, 28, 1))

    # Block 1
    # Path 1: 1x1 pooling
    path1 = layers.MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    path1 = layers.Flatten()(path1)
    path1 = layers.Dropout(0.5)(path1)

    # Path 2: 2x2 pooling
    path2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    path2 = layers.Flatten()(path2)
    path2 = layers.Dropout(0.5)(path2)

    # Path 3: 4x4 pooling
    path3 = layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)
    path3 = layers.Flatten()(path3)
    path3 = layers.Dropout(0.5)(path3)

    # Concatenate outputs of all paths
    block1_output = layers.Concatenate()([path1, path2, path3])
    
    # Fully connected layer after Block 1
    fc_layer = layers.Dense(128, activation='relu')(block1_output)

    # Reshape for Block 2
    reshaped_output = layers.Reshape((-1, 1, 128))(fc_layer)  # Adjust to 4D tensor

    # Block 2
    # Path 1: 1x1 Convolution
    path1 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(reshaped_output)

    # Path 2: 1x1 Convolution followed by 1x7 and 7x1 Convolutions
    path2 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(reshaped_output)
    path2 = layers.Conv2D(32, (1, 7), activation='relu', padding='same')(path2)
    path2 = layers.Conv2D(32, (7, 1), activation='relu', padding='same')(path2)

    # Path 3: 1x1 Convolution followed by alternating 7x1 and 1x7 Convolutions
    path3 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(reshaped_output)
    path3 = layers.Conv2D(32, (7, 1), activation='relu', padding='same')(path3)
    path3 = layers.Conv2D(32, (1, 7), activation='relu', padding='same')(path3)
    path3 = layers.Conv2D(32, (7, 1), activation='relu', padding='same')(path3)

    # Path 4: Average Pooling followed by 1x1 Convolution
    path4 = layers.AveragePooling2D(pool_size=(2, 2))(reshaped_output)
    path4 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(path4)

    # Concatenate outputs of all paths
    block2_output = layers.Concatenate(axis=-1)([path1, path2, path3, path4])

    # Flatten and Fully Connected Layers
    flat_output = layers.Flatten()(block2_output)
    fc_output1 = layers.Dense(128, activation='relu')(flat_output)
    fc_output2 = layers.Dense(10, activation='softmax')(fc_output1)  # 10 classes for MNIST

    # Create model
    model = models.Model(inputs=input_layer, outputs=fc_output2)

    return model

# Instantiate the model
model = dl_model()
model.summary()