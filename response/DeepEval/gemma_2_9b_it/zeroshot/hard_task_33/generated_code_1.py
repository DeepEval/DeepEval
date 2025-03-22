from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_img = keras.Input(shape=(28, 28, 1))

    # Define the block
    def block(input_tensor):
        x = layers.Conv2D(32, 1, activation='relu')(input_tensor)
        x = layers.DepthwiseConv2D(3, activation='relu')(x)
        x = layers.Conv2D(32, 1)(x)
        return layers.add([input_tensor, x])

    # Create the three branches
    branch1 = block(input_img)
    branch2 = block(layers.MaxPooling2D(pool_size=(2, 2))(branch1))
    branch3 = block(layers.MaxPooling2D(pool_size=(2, 2))(branch2))

    # Concatenate the branches
    merged = layers.concatenate([branch1, branch2, branch3], axis=-1)

    # Flatten and fully connect
    x = layers.Flatten()(merged)
    output = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = keras.Model(inputs=input_img, outputs=output)

    return model