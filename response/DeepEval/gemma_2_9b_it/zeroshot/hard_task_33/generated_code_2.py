import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    inputs = layers.Input(shape=(28, 28, 1)) 

    # Define the block
    def block(x):
        x = layers.Conv2D(filters=32, kernel_size=1, activation='relu')(x)
        x = layers.DepthwiseConv2D(kernel_size=3, strides=1, activation='relu')(x)
        x = layers.Conv2D(filters=32, kernel_size=1, activation='relu')(x)
        return layers.Add()([x, inputs])  

    # Create the three branches
    branch1 = block(inputs)
    branch2 = block(branch1)
    branch3 = block(branch2)

    # Concatenate branches
    x = layers.Concatenate()([branch1, branch2, branch3])

    # Flatten and classify
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model