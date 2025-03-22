from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_tensor = keras.Input(shape=(32, 32, 3))

    # Branch 1
    branch1 = layers.Conv2D(64, (1, 1))(input_tensor)

    # Branch 2
    branch2 = layers.Conv2D(64, (1, 1))(input_tensor)
    branch2 = layers.Conv2D(64, (3, 3))(branch2)

    # Branch 3
    branch3 = layers.Conv2D(64, (1, 1))(input_tensor)
    branch3 = layers.Conv2D(64, (3, 3))(branch3)
    branch3 = layers.Conv2D(64, (3, 3))(branch3)

    # Concatenate branches
    concat_layer = layers.Concatenate()([branch1, branch2, branch3])

    # Adjust output dimensions
    concat_layer = layers.Conv2D(3, (1, 1))(concat_layer)

    # Fuse with direct input connection
    main_path = layers.Add()([concat_layer, input_tensor])

    # Classify
    x = layers.Flatten()(main_path)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    output_layer = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=input_tensor, outputs=output_layer)
    
    return model