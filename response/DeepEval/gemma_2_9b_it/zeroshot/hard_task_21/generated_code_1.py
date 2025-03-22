import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    
    input_tensor = layers.Input(shape=(32, 32, 3))  

    # Main Path
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=2))(input_tensor) 
    
    # 1x1 branch
    x1 = layers.Conv2D(64, 1, padding='same', activation='relu')(x[0])
    
    # 3x3 branch
    x2 = layers.Conv2D(64, 3, padding='same', activation='relu')(x[1])
    
    # 5x5 branch
    x3 = layers.Conv2D(64, 5, padding='same', activation='relu')(x[2])

    x = layers.concatenate([x1, x2, x3], axis=3) 

    # Branch Path
    branch = layers.Conv2D(64, 1, padding='same', activation='relu')(input_tensor)

    # Add main and branch paths
    x = layers.Add()([x, branch])

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    output_tensor = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
    return model