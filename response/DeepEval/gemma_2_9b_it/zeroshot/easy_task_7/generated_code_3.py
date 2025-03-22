import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    input_layer = layers.Input(shape=(28, 28, 1)) 

    # Main Path
    x = layers.Conv2D(32, kernel_size=3, activation='relu')(input_layer)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(64, kernel_size=3, activation='relu')(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(32, kernel_size=3, activation='relu')(x)  

    # Branch Path
    branch_x = layers.Conv2D(32, kernel_size=1, activation='relu')(input_layer) 

    # Combine Outputs
    x = layers.Add()([x, branch_x])

    # Flatten and Output Layer
    x = layers.Flatten()(x)
    output_layer = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model