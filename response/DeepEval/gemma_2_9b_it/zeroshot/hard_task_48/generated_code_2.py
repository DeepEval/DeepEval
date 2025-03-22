import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    
    model = tf.keras.models.Sequential()

    # Block 1
    model.add(layers.Lambda(lambda x: tf.split(x, 3, axis=3), input_shape=(32, 32, 3)))  
    for i in range(3):
        model.add(layers.Conv2D(32, kernel_size=(i+1)*1, activation='relu'))
        model.add(layers.BatchNormalization()) 
    model.add(layers.Lambda(lambda x: tf.concat(x, axis=3)))

    # Block 2
    input_tensor = model.output
    
    # Path 1
    path1 = layers.Conv2D(32, kernel_size=1, activation='relu')(input_tensor)

    # Path 2
    path2 = layers.AveragePooling2D(pool_size=(3, 3))(input_tensor)
    path2 = layers.Conv2D(32, kernel_size=1, activation='relu')(path2)

    # Path 3
    path3 = layers.Conv2D(32, kernel_size=1, activation='relu')(input_tensor)
    path3_1 = layers.Conv2D(32, kernel_size=(1, 3), activation='relu')(path3)
    path3_2 = layers.Conv2D(32, kernel_size=(3, 1), activation='relu')(path3)
    path3 = tf.concat([path3_1, path3_2], axis=3)

    # Path 4
    path4 = layers.Conv2D(32, kernel_size=1, activation='relu')(input_tensor)
    path4 = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(path4)
    path4_1 = layers.Conv2D(32, kernel_size=(1, 3), activation='relu')(path4)
    path4_2 = layers.Conv2D(32, kernel_size=(3, 1), activation='relu')(path4)
    path4 = tf.concat([path4_1, path4_2], axis=3)

    # Concatenate all paths
    output_tensor = tf.concat([path1, path2, path3, path4], axis=3)

    # Final layers
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))

    return model