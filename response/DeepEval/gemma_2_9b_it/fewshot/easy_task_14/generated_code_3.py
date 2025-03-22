import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    x = GlobalAveragePooling2D()(input_layer)
    x = Dense(units=32, activation='relu')(x)
    x = Dense(units=32, activation='relu')(x)
    
    # Reshape to align with input shape
    x = Reshape((32, 32, 3))(x)
    
    # Element-wise multiplication
    x = keras.layers.multiply([input_layer, x]) 
    
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model