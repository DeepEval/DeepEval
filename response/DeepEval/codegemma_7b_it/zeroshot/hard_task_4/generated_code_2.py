import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer
    inputs = keras.Input(shape=(32, 32, 3))

    # Increase dimensionality of input channels with a 1x1 convolution
    x = layers.Conv2D(32, (1, 1), padding='same')(inputs)

    # Extract initial features using a 3x3 depthwise separable convolution
    x = layers.DepthwiseConv2D(3, padding='same')(x)

    # Compute channel attention weights
    attention = layers.GlobalAveragePooling2D()(x)
    attention = layers.Dense(64, activation='relu')(attention)
    attention = layers.Dense(32, activation='sigmoid')(attention)

    # Reshape attention weights to match initial features
    attention = layers.Reshape((32, 32, 32))(attention)

    # Channel attention weighting
    x = tf.multiply(x, attention)

    # Reduce dimensionality and combine with initial input
    x = layers.Conv2D(32, (1, 1), padding='same')(x)
    x = layers.Add()([x, inputs])

    # Flattening and fully connected layers for classification
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])