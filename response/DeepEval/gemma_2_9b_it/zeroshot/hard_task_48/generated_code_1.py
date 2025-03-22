import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    model = tf.keras.models.Sequential()

    # Block 1
    model.add(layers.Lambda(lambda x: tf.split(x, 3, axis=3))) 
    model.add(layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Lambda(lambda x: tf.concat(x, axis=3)))

    # Block 2
    model.add(layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu'))
    
    # Path 1
    path1 = tf.keras.models.Sequential()
    path1.add(layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu'))
    
    # Path 2
    path2 = tf.keras.models.Sequential()
    path2.add(layers.AveragePooling2D(pool_size=(3, 3)))
    path2.add(layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu'))
    
    # Path 3
    path3 = tf.keras.models.Sequential()
    path3.add(layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu'))
    path3.add(layers.Conv2D(filters=64, kernel_size=(1, 3), activation='relu'))
    path3.add(layers.Conv2D(filters=64, kernel_size=(3, 1), activation='relu'))
    path3.add(layers.Lambda(lambda x: tf.concat(x, axis=3)))
    
    # Path 4
    path4 = tf.keras.models.Sequential()
    path4.add(layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu'))
    path4.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    path4.add(layers.Conv2D(filters=64, kernel_size=(1, 3), activation='relu'))
    path4.add(layers.Conv2D(filters=64, kernel_size=(3, 1), activation='relu'))
    path4.add(layers.Lambda(lambda x: tf.concat(x, axis=3)))
    
    model.add(layers.Concatenate())
    model.add(layers.Flatten())
    model.add(layers.Dense(units=10, activation='softmax'))

    return model