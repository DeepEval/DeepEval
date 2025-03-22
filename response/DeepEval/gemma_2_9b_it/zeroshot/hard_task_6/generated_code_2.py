import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    input_tensor = layers.Input(shape=(32, 32, 3))  

    # Block 1
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor) 
    x = [layers.Conv2D(filters=32 // 3, kernel_size=1, activation='relu')(xi) for xi in x]
    x = layers.Concatenate(axis=-1)(x)

    # Block 2 (Channel Shuffling)
    shape_tensor = layers.Lambda(lambda x: tf.shape(x))(x)
    x = layers.Reshape(target_shape=(shape_tensor[1], shape_tensor[2], 3, 32 // 3))(x) 
    x = layers.Permute(axes=[1, 2, 4, 3])(x)  
    x = layers.Reshape(target_shape=(shape_tensor[1], shape_tensor[2], 32))(x) 

    # Block 3 (Depthwise Separable Convolution)
    x = layers.Conv2D(filters=64, kernel_size=3, activation='relu', depth_wise=True, padding='same')(x)

    # Branch Path
    branch_x = layers.AveragePooling2D(pool_size=(8, 8))(input_tensor)
    branch_x = layers.Flatten()(branch_x)

    # Concatenation
    x = layers.Concatenate()([x, branch_x])

    # Fully Connected Layer
    output_tensor = layers.Dense(units=10, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
    return model