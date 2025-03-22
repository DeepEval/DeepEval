import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    
    input_layer = keras.Input(shape=(32, 32, 3))

    # Main Path
    block1 = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    block1 = layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1)
    block1 = layers.Concatenate()(block1)
    
    block2 = layers.Lambda(lambda x: tf.constant([3, 3, 3], dtype='int32'))(block1)
    block2 = layers.Reshape((-1, 3, 32))(block2)
    block2 = layers.Permute((2, 3, 1))(block2)
    block2 = layers.Reshape((-1, 32, 3))(block2)
    
    block3 = layers.SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2)

    # Branch Path
    branch = layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    branch = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch)
    
    # Concatenate the main path and branch path
    output_tensor = layers.Concatenate()([block3, branch])
    
    # Final Fully Connected Layer
    output = layers.Flatten()(output_tensor)
    output = layers.Dense(units=64, activation='relu')(output)
    output = layers.Dense(units=10, activation='softmax')(output)

    model = keras.Model(inputs=input_layer, outputs=output)

    return model