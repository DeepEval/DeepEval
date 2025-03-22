from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 64))

    # Main Path
    x = layers.Conv2D(filters=32, kernel_size=(1, 1))(inputs)
    
    # Parallel Convolutional Layers
    branch_1 = layers.Conv2D(filters=32, kernel_size=(1, 1))(x)
    branch_2 = layers.Conv2D(filters=32, kernel_size=(3, 3))(x)
    
    main_path_output = layers.concatenate([branch_1, branch_2])

    # Branch Path
    branch_input = layers.Conv2D(filters=32, kernel_size=(3, 3))(inputs)
    branch_path_output = branch_input

    # Combine Outputs
    combined_output = layers.add([main_path_output, branch_path_output])

    # Flatten and Fully Connected Layers
    x = layers.Flatten()(combined_output)
    x = layers.Dense(units=64, activation='relu')(x)
    outputs = layers.Dense(units=10, activation='softmax')(x)  # Assuming 10 classes

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Example Usage
model = dl_model()
model.summary()