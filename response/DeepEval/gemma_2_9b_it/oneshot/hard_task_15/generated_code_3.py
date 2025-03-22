import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Conv2D, Add

def dl_model():  
    
    input_layer = Input(shape=(32, 32, 3))  

    # Main Path
    x = GlobalAveragePooling2D()(input_layer)
    x = Dense(32, activation='relu')(x) 
    x = Dense(3 * 32 * 32, activation='relu')(x)  
    x = keras.layers.Reshape((32, 32, 3))(x)  
    x = keras.layers.Multiply()([input_layer, x]) 

    # Branch Path
    branch_x = input_layer

    # Combine Paths
    combined = Add()([x, branch_x])
    
    # Final Layers
    combined = Dense(64, activation='relu')(combined)
    output_layer = Dense(10, activation='softmax')(combined)  

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model