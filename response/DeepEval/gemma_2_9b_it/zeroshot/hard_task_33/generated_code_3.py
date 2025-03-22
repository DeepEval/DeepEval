import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    
    input_layer = layers.Input(shape=(28, 28, 1))  

    # Define the block
    def build_block(input_tensor):
        x = layers.Conv2D(32, 1, activation='relu')(input_tensor)
        x = layers.DepthwiseConv2D(3, activation='relu')(x)
        x = layers.Conv2D(32, 1, activation='relu')(x)
        return layers.Add()([input_tensor, x])

    # Create three branches
    branch1 = build_block(input_layer)
    branch2 = build_block(input_layer)
    branch3 = build_block(input_layer)

    # Concatenate outputs from branches
    x = layers.Concatenate()([branch1, branch2, branch3])

    # Flatten and classify
    x = layers.Flatten()(x)
    output_layer = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model