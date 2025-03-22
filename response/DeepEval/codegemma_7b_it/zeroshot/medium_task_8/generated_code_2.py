import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize image pixel values between 0 and 1
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Create input layer
    inputs = keras.Input(shape=(32, 32, 3))

    # Main path
    main_output = inputs
    main_output = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(main_output)
    main_output = layers.Lambda(lambda x: x[0])(main_output)

    branch_output = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    branch_output = layers.Lambda(lambda x: x[1])(branch_output)
    branch_output = layers.Conv2D(filters=32, kernel_size=1, strides=1, padding="same")(branch_output)

    main_output = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same")(main_output)
    branch_output = layers.concatenate([main_output, branch_output])
    main_output = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same")(branch_output)

    # Branch path
    branch_output = layers.Conv2D(filters=32, kernel_size=1, strides=1, padding="same")(inputs)

    # Fusion layer
    fusion_output = layers.Add()([main_output, branch_output])

    # Classification layer
    outputs = layers.Flatten()(fusion_output)
    outputs = layers.Dense(units=10, activation="softmax")(outputs)

    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile model
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model

# Train and evaluate model
model = dl_model()
model.fit(x_train, y_train, epochs=10)
model.evaluate(x_test, y_test)