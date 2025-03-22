import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Concatenate, Dense, Reshape

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Main path
    x = GlobalAveragePooling2D()(input_layer)
    x = Dense(units=32, activation='relu')(x)
    x = Dense(units=32, activation='relu')(x)
    x = Reshape((32, 32, 3))(x)  

    # Branch path
    branch_path = input_layer

    # Combine paths
    combined = keras.layers.add([x, branch_path])
    
    # Final layers
    combined = Dense(units=16, activation='relu')(combined)
    output_layer = Dense(units=10, activation='softmax')(combined)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model