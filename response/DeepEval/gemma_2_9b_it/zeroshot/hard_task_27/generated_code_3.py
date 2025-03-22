import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    input_layer = layers.Input(shape=(32, 32, 3))  

    # Depthwise separable convolutional layer
    x = layers.Conv2D(filters=128, kernel_size=(7, 7), 
                      depth_multiplier=1, padding='same', activation='relu')(input_layer)
    x = layers.LayerNormalization()(x)

    # Two fully connected layers with channel-wise feature transformation
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x) 
    x = layers.Dense(128, activation='relu')(x) 

    # Combine original input with processed features
    x = layers.Add()([input_layer, x])

    # Final fully connected layers for classification
    x = layers.Flatten()(x)
    output_layer = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return model