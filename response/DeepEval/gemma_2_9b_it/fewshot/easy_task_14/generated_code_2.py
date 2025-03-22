import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    x = GlobalAveragePooling2D()(input_layer)
    
    # Fully connected layers to learn channel correlations
    x = Dense(units=32, activation='relu')(x)
    x = Dense(units=32, activation='relu')(x)
    
    # Reshape and element-wise multiplication
    x = Reshape(target_shape=(32, 32, 3))(x)
    x = keras.layers.multiply([input_layer, x])
    
    # Flatten and final fully connected layer
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model