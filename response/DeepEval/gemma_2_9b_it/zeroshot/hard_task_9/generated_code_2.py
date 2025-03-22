from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_tensor = keras.Input(shape=(32, 32, 3))

    # Branch 1: 1x1 convolution
    branch1 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)

    # Branch 2: 1x1 convolution followed by 3x3 convolution
    branch2 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)
    branch2 = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch2)

    # Branch 3: 1x1 convolution followed by two 3x3 convolutions
    branch3 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)
    branch3 = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch3)
    branch3 = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch3)

    # Concatenate outputs from branches
    concat_output = layers.concatenate([branch1, branch2, branch3])

    # 1x1 convolution to adjust output dimensions
    concat_output = layers.Conv2D(filters=3, kernel_size=(1, 1), activation='relu')(concat_output)

    # Add branch output to main path output
    output = layers.add([input_tensor, concat_output])

    # Fully connected layers
    output = layers.Flatten()(output)
    output = layers.Dense(units=128, activation='relu')(output)
    output = layers.Dense(units=64, activation='relu')(output)
    output = layers.Dense(units=10, activation='softmax')(output)

    model = keras.Model(inputs=input_tensor, outputs=output)
    
    return model