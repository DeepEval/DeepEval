from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    
    # Define the same block
    def same_block(input_tensor):
        x = layers.GlobalAveragePooling2D()(input_tensor)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(input_tensor.shape[-1], activation='relu')(x)
        x = layers.Reshape(input_tensor.shape[1:])(x)
        return layers.multiply([input_tensor, x])

    # Define the input layer
    inputs = keras.Input(shape=(32, 32, 3))

    # Create the two branches
    branch1 = same_block(inputs)
    branch2 = same_block(inputs)

    # Concatenate the outputs of the branches
    x = layers.concatenate([branch1, branch2])

    # Flatten and add a fully connected layer
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model