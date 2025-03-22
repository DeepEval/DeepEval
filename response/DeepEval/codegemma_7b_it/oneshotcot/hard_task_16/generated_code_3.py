import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def residual_block(inputs, filters):
    """A residual block with three convolutional layers."""
    x = layers.Conv2D(filters, (1, 1), kernel_initializer='he_normal', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, (1, 1), kernel_initializer='he_normal', padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Add residual connection
    x = layers.Add()([x, inputs])
    x = layers.Activation('relu')(x)

    return x

def transition_conv(inputs, filters):
    """A transition convolution layer."""
    x = layers.Conv2D(filters, (1, 1), kernel_initializer='he_normal', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    return x

def block_2(inputs):
    """Block 2 with global max pooling."""
    # Main path
    main_path = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inputs)

    # Branch path
    branch = layers.Conv2D(128, (1, 1), kernel_initializer='he_normal', padding='same')(inputs)
    branch = layers.BatchNormalization()(branch)
    branch = layers.Activation('relu')(branch)
    branch = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(branch)

    # Add main path and branch paths
    main_path = layers.Add()([main_path, branch])

    # Generate channel-matching weights
    branch = layers.Conv2D(128, (1, 1), kernel_initializer='he_normal', padding='same')(main_path)
    branch = layers.BatchNormalization()(branch)
    branch = layers.Activation('relu')(branch)
    branch = layers.Flatten()(branch)
    branch = layers.Dense(128, kernel_initializer='he_normal')(branch)
    branch = layers.Dense(main_path.shape[-1], kernel_initializer='he_normal')(branch)
    branch = layers.Reshape((1, 1, main_path.shape[-1]))(branch)

    # Multiply weights with main path
    main_path = layers.multiply([main_path, branch])

    return main_path

def dl_model():
    """The CIFAR-10 image classification model."""
    inputs = keras.Input(shape=(32, 32, 3))

    # Block 1
    x = layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Residual blocks
    for filters in [32, 64, 128]:
        x = residual_block(x, filters)
        x = transition_conv(x, filters)

    # Block 2
    x = block_2(x)

    # Output layer
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model

model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])