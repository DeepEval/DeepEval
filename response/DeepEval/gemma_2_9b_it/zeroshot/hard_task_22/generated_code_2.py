import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    
    input_tensor = layers.Input(shape=(32, 32, 3))

    # Main Path
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=2))(input_tensor) 
    
    # 1x1 branch
    branch_output = layers.Conv2D(filters=64, kernel_size=(1,1))(input_tensor)
    
    # 3x3 branch
    x1 = layers.Conv2D(filters=64, kernel_size=(1,1))(x[0])
    x1 = layers.SeparableConv2D(filters=128, kernel_size=(3,3))(x1) 
    x1 = layers.Activation('relu')(x1)

    # 5x5 branch
    x2 = layers.Conv2D(filters=64, kernel_size=(1,1))(x[1])
    x2 = layers.SeparableConv2D(filters=128, kernel_size=(5,5))(x2)
    x2 = layers.Activation('relu')(x2)
    
    # Combine branches
    x = layers.concatenate([x1, x2, x[2]])  

    # Branch Path
    branch_output = layers.Conv2D(filters=128, kernel_size=(1,1))(branch_output)
    branch_output = layers.Activation('relu')(branch_output)

    # Fusion
    x = layers.Add()([x, branch_output])
    
    # Flatten and Fully Connected Layers
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    

    model = tf.keras.Model(inputs=input_tensor, outputs=outputs)

    return model