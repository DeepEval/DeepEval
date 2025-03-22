import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    
    input_tensor = keras.Input(shape=(28, 28, 1))  

    # Block 1
    x_1x1 = layers.AveragePooling2D((1, 1), strides=(1, 1))(input_tensor)
    x_2x2 = layers.AveragePooling2D((2, 2), strides=(2, 2))(input_tensor)
    x_4x4 = layers.AveragePooling2D((4, 4), strides=(4, 4))(input_tensor)

    x_1x1 = layers.Flatten()(x_1x1)
    x_1x1 = layers.Dropout(0.25)(x_1x1)  
    x_2x2 = layers.Flatten()(x_2x2)
    x_2x2 = layers.Dropout(0.25)(x_2x2)
    x_4x4 = layers.Flatten()(x_4x4)
    x_4x4 = layers.Dropout(0.25)(x_4x4)
    
    merged_block1 = layers.Concatenate()([x_1x1, x_2x2, x_4x4])

    # Fully connected and reshape
    x = layers.Dense(128, activation='relu')(merged_block1)
    x = layers.Reshape((128, 1))(x) 

    # Block 2
    x_1x1_conv = layers.Conv2D(32, (1, 1), activation='relu')(x)
    x_2x2_conv = layers.Conv2D(32, (1, 1), activation='relu')(x)
    x_2x2_conv = layers.Conv2D(64, (3, 3), activation='relu')(x_2x2_conv)
    x_2x2_conv = layers.Conv2D(64, (3, 3), activation='relu')(x_2x2_conv) 

    x_3x3_conv = layers.AveragePooling2D((2, 2), strides=(2, 2))(x)
    x_3x3_conv = layers.Conv2D(32, (1, 1), activation='relu')(x_3x3_conv)

    merged_block2 = layers.Concatenate()([x_1x1_conv, x_2x2_conv, x_3x3_conv])

    # Flatten and output layers
    x = layers.Flatten()(merged_block2)
    x = layers.Dense(10, activation='softmax')(x) 

    model = keras.Model(inputs=input_tensor, outputs=x)
    return model