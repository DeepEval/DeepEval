import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    inputs = layers.Input(shape=(28, 28, 1))

    # First Block: Average Pooling
    x = layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(inputs)
    x = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(x)

    x = layers.Flatten()(x)
    x = layers.concatenate([x, x, x], axis=-1)  

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Reshape((1, 128, 1))(x)

    # Second Block: Depthwise Separable Convolutions
    x = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=2))(x)
    
    conv_outputs = []
    for i in range(4):
        x = layers.Lambda(lambda x: x[i])(x)
        if i == 0:
            x = layers.Conv2D(32, kernel_size=(1, 1), padding='same', activation='relu')(x)
        elif i == 1:
            x = layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(x)
        elif i == 2:
            x = layers.Conv2D(32, kernel_size=(5, 5), padding='same', activation='relu')(x)
        else:
            x = layers.Conv2D(32, kernel_size=(7, 7), padding='same', activation='relu')(x)
        conv_outputs.append(x)

    x = layers.concatenate(conv_outputs, axis=2)
    x = layers.Flatten()(x)
    
    outputs = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model