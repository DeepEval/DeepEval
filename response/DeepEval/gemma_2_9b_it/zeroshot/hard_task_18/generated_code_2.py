import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_img = keras.Input(shape=(32, 32, 3))

    # First Block
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_img)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.AveragePooling2D((2, 2))(x)

    # Second Block
    main_path = layers.GlobalAveragePooling2D()(x)
    main_path = layers.Dense(64, activation='relu')(main_path)
    main_path = layers.Dense(64, activation='relu')(main_path)
    
    # Reshape channel weights
    channel_weights = layers.Reshape((32, 32, 64))(main_path)

    # Multiply with input features
    output = input_img * channel_weights

    # Flatten and classify
    output = layers.Flatten()(output)
    output = layers.Dense(10, activation='softmax')(output)

    model = keras.Model(inputs=input_img, outputs=output)
    return model

model = dl_model()
print(model.summary())