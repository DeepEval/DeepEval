import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 pixels with 3 color channels
    inputs = layers.Input(shape=input_shape)

    # Branch 1: 1x1 Conv + 3x3 Conv
    branch1 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    branch1 = layers.Conv2D(64, (3, 3), activation='relu')(branch1)
    branch1 = layers.Dropout(0.5)(branch1)

    # Branch 2: 1x1 Conv + 1x7 Conv + 7x1 Conv + 3x3 Conv
    branch2 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    branch2 = layers.Conv2D(64, (1, 7), activation='relu')(branch2)
    branch2 = layers.Conv2D(64, (7, 1), activation='relu')(branch2)
    branch2 = layers.Conv2D(64, (3, 3), activation='relu')(branch2)
    branch2 = layers.Dropout(0.5)(branch2)

    # Branch 3: Max Pooling
    branch3 = layers.MaxPooling2D(pool_size=(2, 2))(inputs)
    branch3 = layers.Dropout(0.5)(branch3)

    # Concatenate the outputs from all branches
    concatenated = layers.concatenate([branch1, branch2, branch3])

    # Fully connected layers
    x = layers.Flatten()(concatenated)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example of using the model
model = dl_model()
model.summary()  # To display the model architecture