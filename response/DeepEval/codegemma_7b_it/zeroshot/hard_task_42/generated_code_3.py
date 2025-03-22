import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_img = keras.Input(shape=(28, 28, 1))

    # Block 1
    pool_outputs = []
    for pool_size, strides in [(1, 1), (2, 2), (4, 4)]:
        pool_outputs.append(
            layers.MaxPooling2D(pool_size=pool_size, strides=strides)(input_img)
        )

    # Flatten and regularize pooling outputs
    pool_outputs = [layers.Flatten()(pool_out) for pool_out in pool_outputs]
    pool_outputs = [
        layers.Dropout(0.2)(x) for x in pool_outputs
    ]

    # Concatenate pooling outputs
    concat_pool_outputs = layers.concatenate(pool_outputs)

    # Fully connected layer and reshape
    fc_outputs = layers.Dense(500, activation='relu')(concat_pool_outputs)
    fc_outputs = layers.Reshape((-1, 1, 1))(fc_outputs)

    # Block 2
    path1 = layers.Conv2D(64, (1, 1), padding='same')(fc_outputs)
    path2 = layers.Conv2D(64, (1, 1), padding='same')(fc_outputs)
    path2 = layers.Conv2D(64, (1, 7), padding='same')(path2)
    path2 = layers.Conv2D(64, (7, 1), padding='same')(path2)
    path3 = layers.Conv2D(64, (1, 1), padding='same')(fc_outputs)
    path3 = layers.Conv2D(64, (7, 1), padding='same')(path3)
    path3 = layers.Conv2D(64, (1, 7), padding='same')(path3)
    path3 = layers.Conv2D(64, (7, 1), padding='same')(path3)
    path4 = layers.AveragePooling2D((2, 2), strides=(2, 2))(fc_outputs)
    path4 = layers.Conv2D(64, (1, 1), padding='same')(path4)

    # Concatenate paths
    concat_path_outputs = layers.concatenate([path1, path2, path3, path4])

    # Final fully connected layers
    flatten_outputs = layers.Flatten()(concat_path_outputs)
    output = layers.Dense(10, activation='softmax')(flatten_outputs)

    # Create model
    model = Model(inputs=input_img, outputs=output)
    return model

# Create model instance
model = dl_model()