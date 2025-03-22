import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 with 3 color channels
    inputs = layers.Input(shape=input_shape)

    # Main Path
    x = layers.Conv2D(16, (1, 1), activation='relu')(inputs)

    # Splitting into three branches
    # Branch 1: 3x3 Convolution for local feature extraction
    branch1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)

    # Branch 2: Downsample -> 3x3 Conv -> Upsample
    branch2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
    branch2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(branch2)
    branch2 = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(branch2)

    # Branch 3: Downsample -> 3x3 Conv -> Upsample
    branch3 = layers.AveragePooling2D(pool_size=(2, 2))(x)
    branch3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(branch3)
    branch3 = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(branch3)

    # Concatenate the outputs of the three branches
    concatenated = layers.concatenate([branch1, branch2, branch3])

    # 1x1 Convolution to form the main path output
    main_output = layers.Conv2D(64, (1, 1), activation='relu')(concatenated)

    # Branch Path: 1x1 Convolution to match channels
    branch_output = layers.Conv2D(64, (1, 1), activation='relu')(inputs)

    # Fuse main path and branch path outputs
    fused_output = layers.add([main_output, branch_output])

    # Global Average Pooling and Fully Connected Layer for classification
    x = layers.GlobalAveragePooling2D()(fused_output)
    outputs = layers.Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10

    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example of how to create the model
model = dl_model()
model.summary()  # Display the model architecture