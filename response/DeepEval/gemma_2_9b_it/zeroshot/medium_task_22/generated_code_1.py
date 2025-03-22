from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_img = keras.Input(shape=(32, 32, 3))

    # Branch 1: 3x3 convolutions
    x_branch1 = layers.Conv2D(32, (3, 3), activation='relu')(input_img)
    x_branch1 = layers.Conv2D(64, (3, 3), activation='relu')(x_branch1)
    
    # Branch 2: 1x1 convolutions followed by two 3x3 convolutions
    x_branch2 = layers.Conv2D(64, (1, 1), activation='relu')(input_img)
    x_branch2 = layers.Conv2D(64, (3, 3), activation='relu')(x_branch2)
    x_branch2 = layers.Conv2D(128, (3, 3), activation='relu')(x_branch2)

    # Branch 3: Max pooling
    x_branch3 = layers.MaxPooling2D((2, 2))(input_img)
    x_branch3 = layers.Conv2D(64, (3, 3), activation='relu')(x_branch3)

    # Concatenate outputs from the three branches
    x_merged = layers.concatenate([x_branch1, x_branch2, x_branch3])

    # Flatten and feed into fully connected layers
    x_flat = layers.Flatten()(x_merged)
    x = layers.Dense(128, activation='relu')(x_flat)
    output = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=input_img, outputs=output)
    return model