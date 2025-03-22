import keras
from keras import layers

def dl_model():

    # Input layer
    inputs = layers.Input(shape=(28, 28, 1))

    # Block 1
    x1 = layers.AveragePooling2D(pool_size=1, strides=1)(inputs)
    x1 = layers.Flatten()(x1)
    x1 = layers.Dropout(0.5)(x1)

    x2 = layers.AveragePooling2D(pool_size=2, strides=2)(inputs)
    x2 = layers.Flatten()(x2)
    x2 = layers.Dropout(0.5)(x2)

    x3 = layers.AveragePooling2D(pool_size=4, strides=4)(inputs)
    x3 = layers.Flatten()(x3)
    x3 = layers.Dropout(0.5)(x3)

    # Concatenate outputs from parallel paths
    concat_path = keras.layers.concatenate([x1, x2, x3])

    # Block 2
    branch1 = layers.Conv2D(filters=32, kernel_size=1, activation='relu')(inputs)

    branch2 = layers.Conv2D(filters=32, kernel_size=1, activation='relu')(inputs)
    branch2 = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(branch2)

    branch3 = layers.Conv2D(filters=32, kernel_size=1, activation='relu')(inputs)
    branch3 = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(branch3)
    branch3 = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(branch3)

    branch4 = layers.AveragePooling2D(pool_size=2, strides=2)(inputs)
    branch4 = layers.Conv2D(filters=32, kernel_size=1, activation='relu')(branch4)

    # Concatenate outputs from branch connections
    concat_branch = keras.layers.concatenate([branch1, branch2, branch3, branch4])

    # Fully connected layers for classification
    flatten_branch = layers.Flatten()(concat_branch)
    dense1 = layers.Dense(500, activation='relu')(flatten_branch)
    dense2 = layers.Dense(10, activation='softmax')(dense1)

    # Create model
    model = keras.Model(inputs=inputs, outputs=dense2)

    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])