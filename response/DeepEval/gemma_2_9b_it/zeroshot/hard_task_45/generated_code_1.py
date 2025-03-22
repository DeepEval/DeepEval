import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    model = models.Sequential()

    # First Block
    model.add(layers.Lambda(lambda x: tf.split(x, 3, axis=-1)))  
    for i in range(3):
        model.add(layers.Conv2D(filters=32, kernel_size=(i+1, i+1), 
                               activation='relu', padding='same'))
    model.add(layers.Lambda(lambda x: tf.concat(x, axis=-1))) 

    # Second Block
    x = model.output
    branch1 = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x)
    branch2 = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x)
    branch2 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch2)
    branch3 = layers.MaxPool2D((2, 2))(x)
    branch3 = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(branch3)
    
    branches = [branch1, branch2, branch3]
    x = tf.concat(branches, axis=-1)

    # Final Layers
    x = layers.Flatten()(x)
    x = layers.Dense(10, activation='softmax')(x)

    model = models.Model(inputs=model.input, outputs=x)
    return model