from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_shape = (32, 32, 3)  
    
    # Input Layer
    inputs = keras.Input(shape=input_shape)

    # Split the input into three groups
    split_outputs = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)

    # Group 1: 1x1 convolutions
    conv1 = layers.Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(split_outputs[0])

    # Group 2: 3x3 convolutions
    conv2 = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(split_outputs[1])

    # Group 3: 5x5 convolutions
    conv3 = layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(split_outputs[2])

    # Concatenate the outputs from each group
    merged = layers.Concatenate(axis=-1)([conv1, conv2, conv3])

    # Flatten and dense layer
    flatten = layers.Flatten()(merged)
    outputs = layers.Dense(10, activation='softmax')(flatten)

    # Build the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model