import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    model = tf.keras.models.Sequential()

    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # Block 1: Splitting and Depthwise Separable Convolutions
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # Group 1
    x1 = layers.Conv2D(32, (1, 1), activation='relu', name='dw_conv_1x1_1')(x[0])
    x1 = layers.DepthwiseConv2D((3, 3), strides=1, padding='same', activation='relu', name='dw_conv_3x3_1')(x1)

    # Group 2
    x2 = layers.Conv2D(32, (1, 1), activation='relu', name='dw_conv_1x1_2')(x[1])
    x2 = layers.DepthwiseConv2D((5, 5), strides=1, padding='same', activation='relu', name='dw_conv_5x5_2')(x2)

    # Group 3
    x3 = layers.Conv2D(32, (1, 1), activation='relu', name='dw_conv_1x1_3')(x[2])
    x3 = layers.DepthwiseConv2D((3, 3), strides=1, padding='same', activation='relu', name='dw_conv_3x3_3')(x3)

    x = layers.Concatenate(axis=-1)([x1, x2, x3])

    # Block 2: Multi-Branch Feature Extraction
    branch1 = layers.Conv2D(64, (1, 1), activation='relu')(x)
    branch2 = layers.Conv2D(64, (1, 1), activation='relu')(x)
    branch2 = layers.Conv2D(64, (3, 3), activation='relu')(branch2)
    branch3 = layers.MaxPooling2D((2, 2))(x)
    branch3 = layers.Conv2D(64, (1, 1), activation='relu')(branch3)

    x = layers.Concatenate(axis=-1)([branch1, branch2, branch3])

    # Output Layer
    x = layers.Flatten()(x)
    output_layer = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    return model