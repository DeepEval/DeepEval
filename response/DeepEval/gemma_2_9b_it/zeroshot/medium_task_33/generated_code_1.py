import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    input_img = layers.Input(shape=(32, 32, 3))  

    # Split the input image into three channels
    x = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=2))(input_img) 

    # Feature extraction using separable convolutions
    x_1x1 = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x[0])
    x_3x3 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x[1])
    x_5x5 = layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(x[2])

    # Concatenate the outputs
    x = layers.Concatenate(axis=3)([x_1x1, x_3x3, x_5x5])

    # Fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(units=128, activation='relu')(x)
    x = layers.Dense(units=64, activation='relu')(x)
    output = layers.Dense(units=10, activation='softmax')(x)  

    model = tf.keras.Model(inputs=input_img, outputs=output)
    return model