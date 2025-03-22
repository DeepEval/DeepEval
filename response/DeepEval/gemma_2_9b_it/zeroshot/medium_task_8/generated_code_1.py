import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    input_tensor = layers.Input(shape=(32, 32, 3))  

    # Main Path
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor) 
    
    # Group 1 remains unchanged
    branch1 = x[0]

    # Group 2: Feature Extraction
    branch2 = layers.Conv2D(32, (3, 3), activation='relu')(x[1]) 

    # Group 3: Combination
    branch3 = layers.Conv2D(32, (3, 3), activation='relu')(x[2])
    
    # Concatenate outputs of all groups
    main_path = layers.concatenate([branch1, branch2, branch3])

    # Branch Path
    branch_path = layers.Conv2D(32, (1, 1), activation='relu')(input_tensor) 

    # Fusion
    output = layers.add([main_path, branch_path])

    # Classification
    output = layers.Flatten()(output)
    output = layers.Dense(10, activation='softmax')(output)

    model = tf.keras.Model(inputs=input_tensor, outputs=output)
    
    return model