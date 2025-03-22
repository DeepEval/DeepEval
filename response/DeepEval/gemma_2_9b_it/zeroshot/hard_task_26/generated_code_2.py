from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))

    # Main Path
    x = layers.Conv2D(filters=16, kernel_size=1, activation='relu')(inputs)
    
    branch1 = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(x)
    branch2 = layers.MaxPooling2D(pool_size=(2, 2))(x)
    branch2 = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(branch2)
    branch2 = layers.UpSampling2D(size=(2, 2))(branch2)

    branch3 = layers.MaxPooling2D(pool_size=(2, 2))(x)
    branch3 = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(branch3)
    branch3 = layers.UpSampling2D(size=(2, 2))(branch3)

    x = layers.concatenate([branch1, branch2, branch3], axis=-1)
    x = layers.Conv2D(filters=16, kernel_size=1, activation='relu')(x)

    # Branch Path
    branch_x = layers.Conv2D(filters=16, kernel_size=1, activation='relu')(inputs)
    
    # ... (Add your branch path layers here) ...

    # Concatenate and classify
    x = layers.add([x, branch_x])
    x = layers.Flatten()(x)
    x = layers.Dense(units=128, activation='relu')(x)
    outputs = layers.Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model