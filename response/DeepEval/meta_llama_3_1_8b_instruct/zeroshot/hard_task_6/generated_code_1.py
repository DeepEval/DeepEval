# Import necessary packages
from tensorflow.keras.layers import (
    Input,
    Lambda,
    Conv2D,
    Activation,
    AveragePooling2D,
    Reshape,
    Permute,
    Concatenate,
    Add,
    DepthwiseConv2D,
    SeparableConv2D,
    Flatten,
    Dense
)
from tensorflow.keras.models import Model
import tensorflow as tf

def dl_model():
    # Define input shape
    input_shape = (32, 32, 3)

    # Define input layer
    inputs = Input(shape=input_shape, name='input_layer')

    # Block 1
    x = Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)
    x = [Conv2D(32, (1, 1), activation='relu')(xi) for xi in x]
    x = Concatenate(axis=-1)(x)
    Block1 = x

    # Block 2
    x = Block1
    h, w, groups, channels_per_group = x.shape[1], x.shape[2], 3, 32 // 3
    x = Reshape((h, w, groups, channels_per_group))(x)
    x = Permute((2, 3, 1, 4))(x)
    x = Reshape((h, w, groups * channels_per_group))(x)
    Block2 = x

    # Block 3
    x = SeparableConv2D(64, (3, 3), activation='relu', padding='same')(Block2)
    Block3 = x

    # Repeat Block 1
    x = Lambda(lambda x: tf.split(x, 3, axis=-1))(Block3)
    x = [Conv2D(32, (1, 1), activation='relu')(xi) for xi in x]
    x = Concatenate(axis=-1)(x)
    Block4 = x

    # Average pooling branch
    avg_pool = AveragePooling2D((2, 2))(inputs)
    avg_pool = Lambda(lambda x: tf.split(x, 3, axis=-1))(avg_pool)
    avg_pool = [Conv2D(32, (1, 1), activation='relu')(xi) for xi in avg_pool]
    avg_pool = Concatenate(axis=-1)(avg_pool)
    avg_pool = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(avg_pool)
    avg_pool = Lambda(lambda x: tf.tile(x, [1, 1, 1, 32]))(avg_pool)

    # Concatenate main path and branch path
    x = Concatenate()([Block4, avg_pool])

    # Flatten and fully connected layer
    x = Flatten()(x)
    x = Dense(10, activation='softmax')(x)

    # Define model
    model = Model(inputs=inputs, outputs=x, name='CIFAR-10_Classifier')
    return model