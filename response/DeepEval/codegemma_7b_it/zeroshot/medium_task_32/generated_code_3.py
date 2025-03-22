import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():

    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Split input into three groups
    inputs = keras.Input(shape=x_train.shape[1:])
    group1, group2, group3 = layers.Lambda(tf.split, arguments=[3])(inputs)

    # Feature extraction for each group
    group1 = layers.SeparableConv2D(32, (1, 1), padding="same")(group1)
    group1 = layers.Activation("relu")(group1)
    group2 = layers.SeparableConv2D(64, (3, 3), padding="same")(group2)
    group2 = layers.Activation("relu")(group2)
    group3 = layers.SeparableConv2D(128, (5, 5), padding="same")(group3)
    group3 = layers.Activation("relu")(group3)

    # Concatenate and fuse features
    concat_axis = -1
    fuse_features = layers.concatenate([group1, group2, group3], axis=concat_axis)

    # Flattening and classification
    fuse_features = layers.Flatten()(fuse_features)
    outputs = layers.Dense(10, activation="softmax")(fuse_features)

    # Create model
    model = keras.Model(inputs, outputs)

    return model