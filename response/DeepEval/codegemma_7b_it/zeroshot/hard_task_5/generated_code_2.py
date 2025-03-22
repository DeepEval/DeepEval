from tensorflow import keras
from tensorflow.keras.layers import (
    Input,
    Lambda,
    Reshape,
    Permute,
    Concatenate,
    Conv2D,
    BatchNormalization,
    Activation,
    ZeroPadding2D,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dense,
)
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.regularizers import l2

def dl_model():
    """
    Builds a deep learning model for image classification using the CIFAR-10 dataset.

    Returns:
        A Keras Model object.
    """

    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize the input data
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Define the input layer
    input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
    inputs = Input(shape=input_shape)

    # Block 1
    group1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(inputs)
    group1 = [
        Conv2D(
            filters=int(32 / 3),
            kernel_size=(1, 1),
            padding="same",
            kernel_initializer=he_normal(),
            kernel_regularizer=l2(l=0.0001),
        )(group)
        for group in group1
    ]
    group1 = [
        BatchNormalization()(conv) for conv in group1
    ]
    group1 = [Activation("relu")(conv) for conv in group1]
    group1 = [
        Conv2D(
            filters=32,
            kernel_size=(1, 1),
            padding="same",
            kernel_initializer=he_normal(),
            kernel_regularizer=l2(l=0.0001),
        )(conv)
        for conv in group1
    ]
    group1 = [
        BatchNormalization()(conv) for conv in group1
    ]
    group1 = [Activation("relu")(conv) for conv in group1]
    group1 = [
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv) for conv in group1
    ]
    group1 = [Lambda(tf.concat, axis=3)(group) for group in zip(*group1)]
    group1_output = Lambda(tf.concat, axis=3)(group1)

    # Block 2
    group2 = Reshape(target_shape=(group1_output.shape[1], group1_output.shape[2], 3, 32))(
        group1_output
    )
    group2 = Permute(dims=(1, 2, 4, 3))(group2)
    group2 = Reshape(target_shape=(group1_output.shape[1], group1_output.shape[2], 3 * 32))(
        group2
    )
    group2 = Activation("relu")(group2)
    group2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(group2)

    # Block 3
    group3 = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer=he_normal(),
        kernel_regularizer=l2(l=0.0001),
        use_bias=False,
    )(group2)
    group3 = BatchNormalization()(group3)
    group3 = Activation("relu")(group3)

    # Branch
    branch = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer=he_normal(),
        kernel_regularizer=l2(l=0.0001),
        use_bias=False,
    )(inputs)
    branch = BatchNormalization()(branch)
    branch = Activation("relu")(branch)

    # Combine main path and branch
    combined = Add()([branch, group3])
    combined = Activation("relu")(combined)

    # Output layer
    outputs = GlobalAveragePooling2D()(combined)
    outputs = Dense(
        units=10,
        kernel_initializer=he_normal(),
        kernel_regularizer=l2(l=0.0001),
        activation="softmax",
    )(outputs)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model