import keras
from keras.layers import Input, GlobalAveragePooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    x = GlobalAveragePooling2D()(input_layer)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    
    # Branch path
    branch = input_layer
    
    # Add the main and branch paths
    x = keras.layers.concatenate([x, branch])
    
    # Final output
    x = Dense(10, activation='softmax')(x)
    
    model = keras.Model(inputs=input_layer, outputs=x)
    
    return model