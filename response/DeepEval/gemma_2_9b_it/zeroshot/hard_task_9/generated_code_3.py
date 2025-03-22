import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    input_tensor = layers.Input(shape=(32, 32, 3))

    # Branch directly connected to the input
    branch_1 = layers.Conv2D(filters=32, kernel_size=1, activation='relu')(input_tensor)

    # Main path
    branch_2 = layers.Conv2D(filters=32, kernel_size=1, activation='relu')(input_tensor)
    branch_2 = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(branch_2)
    branch_2 = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(branch_2)
    
    branch_3 = layers.Conv2D(filters=32, kernel_size=1, activation='relu')(input_tensor)
    branch_3 = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(branch_3)
    branch_3 = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(branch_3)
    
    # Concatenate the outputs of the three branches
    concatenated = layers.Concatenate()([branch_1, branch_2, branch_3])
    
    # Adjust the output dimensions
    concatenated = layers.Conv2D(filters=3, kernel_size=1, activation='relu')(concatenated)

    # Fuse the main path and the branch through addition
    output = layers.Add()([branch_1, concatenated])

    # Classification layers
    output = layers.Flatten()(output)
    output = layers.Dense(units=128, activation='relu')(output)
    output = layers.Dense(units=64, activation='relu')(output)
    output = layers.Dense(units=10, activation='softmax')(output)

    model = models.Model(inputs=input_tensor, outputs=output)

    return model