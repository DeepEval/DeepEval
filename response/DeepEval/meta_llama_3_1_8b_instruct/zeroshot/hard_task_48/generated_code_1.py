import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer
    inputs = keras.Input(shape=(32, 32, 3), name='input_layer')

    # Block 1
    block1 = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    block1_1 = layers.SeparableConv2D(32, (1, 1), activation='relu', name='block1_1')(block1[0])
    block1_1 = layers.BatchNormalization()(block1_1)
    block1_2 = layers.SeparableConv2D(32, (3, 3), activation='relu', name='block1_2')(block1[1])
    block1_2 = layers.BatchNormalization()(block1_2)
    block1_3 = layers.SeparableConv2D(32, (5, 5), activation='relu', name='block1_3')(block1[2])
    block1_3 = layers.BatchNormalization()(block1_3)
    block1 = layers.Concatenate()([block1_1, block1_2, block1_3])

    # Block 2
    block2_1 = layers.SeparableConv2D(32, (1, 1), activation='relu', name='block2_1')(block1)
    block2_2 = layers.AveragePooling2D((3, 3), strides=2, name='block2_2')(block1)
    block2_2 = layers.SeparableConv2D(32, (1, 1), activation='relu', name='block2_3')(block2_2)
    block2_3 = layers.SeparableConv2D(32, (1, 1), activation='relu', name='block2_4')(block1)
    block2_3 = layers.Conv2D(32, (1, 3), activation='relu', name='block2_5')(block2_3)
    block2_3 = layers.Conv2D(32, (3, 1), activation='relu', name='block2_6')(block2_3)
    block2_4 = layers.SeparableConv2D(32, (1, 1), activation='relu', name='block2_7')(block1)
    block2_4 = layers.Conv2D(32, (3, 3), activation='relu', name='block2_8')(block2_4)
    block2_4 = layers.Conv2D(32, (1, 3), activation='relu', name='block2_9')(block2_4)
    block2_4 = layers.Conv2D(32, (3, 1), activation='relu', name='block2_10')(block2_4)
    block2 = layers.Concatenate()([block2_1, block2_2, block2_3, block2_4])

    # Flatten and fully connected
    x = layers.Flatten()(block2)
    outputs = layers.Dense(10, activation='softmax', name='output_layer')(x)

    # Model definition
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Usage
model = dl_model()
model.summary()