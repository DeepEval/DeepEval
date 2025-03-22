import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    input_tensor = layers.Input(shape=(32, 32, 3))

    # Main Path
    x = layers.Conv2D(filters=64, kernel_size=(1, 1))(input_tensor)  
    
    branch1 = layers.Conv2D(filters=64, kernel_size=(1, 7))(x)
    branch1 = layers.Conv2D(filters=64, kernel_size=(7, 1))(branch1)
    
    x = layers.concatenate([x, branch1], axis=-1)
    x = layers.Conv2D(filters=3, kernel_size=(1, 1))(x)  

    # Branch
    branch2 = layers.Conv2D(filters=3, kernel_size=(1, 1))(input_tensor)

    # Merge
    x = layers.add([x, branch2])

    x = layers.Flatten()(x)
    x = layers.Dense(units=128, activation='relu')(x)
    output = layers.Dense(units=10, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_tensor, outputs=output)
    
    return model