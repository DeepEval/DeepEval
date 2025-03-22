import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    model = keras.Sequential()

    # Block 1
    model.add(layers.Conv2D(filters=3, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))

    # Path 1
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(3))  

    # Path 2
    model.add(layers.GlobalMaxPooling2D())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(3))

    # Channel Attention
    model.add(layers.Add())  
    model.add(layers.Activation('sigmoid'))

    # Block 2
    x = model.output

    # Spatial Feature Extraction
    avg_pool = layers.AveragePooling2D(pool_size=(2, 2))(x)
    max_pool = layers.MaxPooling2D(pool_size=(2, 2))(x)
    concat_features = layers.concatenate([avg_pool, max_pool], axis=-1)
    
    model.add(layers.Conv2D(filters=3, kernel_size=(1, 1))(concat_features)) 
    model.add(layers.Activation('sigmoid'))

    # Element-wise Multiplication
    model.add(layers.Multiply()([x, model.output]))

    # Additional Branch
    model.add(layers.Conv2D(filters=3, kernel_size=(1, 1), activation='relu')) 

    # Final Output
    model.add(layers.Add())
    model.add(layers.Activation('relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))

    return model