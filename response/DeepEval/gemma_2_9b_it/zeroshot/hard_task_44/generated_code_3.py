import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    inputs = layers.Input(shape=(32, 32, 3)) 

    # Block 1
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=2))(inputs)
    x = [layers.Conv2D(32, kernel_size=k, activation='relu')(branch) for k, branch in zip([1, 3, 5], x)]
    x = [layers.Dropout(0.25)(branch) for branch in x]
    x = layers.Concatenate()(x)

    # Block 2
    branch1 = layers.Conv2D(64, kernel_size=1, activation='relu')(x)
    branch2 = layers.Conv2D(64, kernel_size=1, activation='relu')(x)
    branch2 = layers.Conv2D(64, kernel_size=5, activation='relu')(branch2)
    branch3 = layers.MaxPooling2D(pool_size=(3, 3))(x)
    branch3 = layers.Conv2D(64, kernel_size=1, activation='relu')(branch3)

    x = layers.Concatenate()([branch1, branch2, branch3])

    # Output
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model