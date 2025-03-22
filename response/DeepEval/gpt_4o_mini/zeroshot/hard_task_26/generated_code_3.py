import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 pixels with 3 color channels
    inputs = layers.Input(shape=input_shape)

    # Main Path
    x = layers.Conv2D(32, (1, 1), padding='same')(inputs)

    # Branch 1
    branch1 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)

    # Branch 2
    branch2 = layers.MaxPooling2D(pool_size=(2, 2))(x)
    branch2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(branch2)
    branch2 = layers.UpSampling2D(size=(2, 2))(branch2)

    # Branch 3
    branch3 = layers.MaxPooling2D(pool_size=(2, 2))(x)
    branch3 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(branch3)
    branch3 = layers.UpSampling2D(size=(2, 2))(branch3)

    # Concatenate all branches
    merged = layers.concatenate([branch1, branch2, branch3])

    # Final 1x1 convolution in main path
    main_output = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(merged)

    # Branch Path
    branch_output = layers.Conv2D(32, (1, 1), padding='same')(inputs)

    # Combine main path and branch path
    combined = layers.add([main_output, branch_output])

    # Flatten and fully connected layers for classification
    x = layers.Flatten()(combined)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10

    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Create the model
model = dl_model()
model.summary()  # Display the model architecture