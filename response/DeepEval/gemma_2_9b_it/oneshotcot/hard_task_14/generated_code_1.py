import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, BatchNormalization, Flatten, Dense, Add

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  

    # Main Path
    x = GlobalAveragePooling2D()(input_layer)
    x = Dense(32, activation='relu')(x) 
    x = Dense(32, activation='relu')(x)  
    x = keras.layers.Reshape((32, 32, 3))(x)  

    # Branch Path
    branch_x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)

    # Add the outputs
    x = Add()([x, branch_x])

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    output_layer = Dense(10, activation='softmax')(x)  

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model