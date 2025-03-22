import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    input_layer = layers.Input(shape=(32, 32, 3))  

    # Depthwise separable convolution with layer normalization
    x = layers.SeparableConv2D(filters=64, kernel_size=(7, 7), activation='relu', padding='same')(input_layer)
    x = layers.LayerNormalization()(x)

    # Two fully connected layers for channel-wise feature transformation
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)

    # Combine original input with processed features
    x = layers.add([input_layer, x])

    # Final classification layers
    x = layers.Flatten()(x)
    output_layer = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return model