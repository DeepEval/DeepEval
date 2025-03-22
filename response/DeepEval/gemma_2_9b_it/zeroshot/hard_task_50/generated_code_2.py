import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_tensor = keras.Input(shape=(32, 32, 3))

    # First Block
    x = layers.MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(input_tensor)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.25)(x)  

    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_tensor)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(input_tensor)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.25)(x)

    x = tf.keras.layers.concatenate([x, x, x]) 

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Reshape((1, 128))(x) 

    # Second Block
    x = layers.Lambda(lambda x: tf.split(x, 4, axis=-1))(x) 
    x = [layers.SeparableConv2D(filters=16, kernel_size=(k, k), activation='relu')(xi) for xi in x for k in [1, 3, 5, 7]]
    x = layers.Concatenate(axis=-1)(x)
    x = layers.Flatten()(x)
    
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=input_tensor, outputs=outputs)
    return model