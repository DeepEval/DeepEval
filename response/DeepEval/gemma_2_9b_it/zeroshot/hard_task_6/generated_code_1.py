import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    input_tensor = layers.Input(shape=(32, 32, 3))  

    # Block 1
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=3))(input_tensor)
    x = [layers.Conv2D(filters=int(tf.shape(input_tensor)[-1]//3), kernel_size=1, activation='relu')(y) for y in x]
    x = layers.Concatenate(axis=3)(x)

    # Block 2
    x = layers.Lambda(lambda x: tf.keras.backend.int_shape(x)[1:])(x)
    x = layers.Reshape((tf.keras.backend.int_shape(x)[0], tf.keras.backend.int_shape(x)[1], 3, tf.keras.backend.int_shape(x)[2]//3))(x)
    x = layers.Permute((2, 3, 1, 0))(x)
    x = layers.Reshape((tf.keras.backend.int_shape(x)[0], tf.keras.backend.int_shape(x)[1], tf.keras.backend.int_shape(x)[2]//3, 3))(x)

    # Block 3
    x = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', activation='relu')(x)

    # Branch path
    branch_x = layers.AveragePooling2D(pool_size=(8, 8))(input_tensor)
    branch_x = layers.Flatten()(branch_x)

    # Concatenate outputs
    concat_x = layers.Concatenate()([x, branch_x])

    # Fully connected layer
    output_tensor = layers.Dense(10, activation='softmax')(concat_x)

    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
    return model