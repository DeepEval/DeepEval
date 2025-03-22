import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_tensor = keras.Input(shape=(32, 32, 3))

    # Block 1
    x = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_tensor)
    x = [layers.Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1))(xi) for xi in x]
    x = [layers.Activation('relu')(xi) for xi in x]
    x = layers.Concatenate(axis=3)(x)

    # Block 2
    x = layers.Lambda(lambda x: tf.keras.backend.shape(x)[-1])(x)  
    x = layers.Reshape((x, x, 3, 1))(x)  
    x = layers.Permute((2, 3, 1, 0))(x)
    x = layers.Reshape((x, x, 1))(x)

    # Block 3
    x = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Branch path
    branch_features = layers.AveragePooling2D((5, 5), strides=(3, 3))(input_tensor)
    branch_features = layers.Conv2D(filters=128, kernel_size=(1, 1))(branch_features)
    branch_features = layers.Flatten()(branch_features)

    # Concatenate
    x = layers.Concatenate()([x, branch_features])

    # Fully connected layers
    x = layers.Dense(128, activation='relu')(x)
    output_tensor = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=input_tensor, outputs=output_tensor)

    return model