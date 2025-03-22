import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model(): 
    
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    x = AveragePooling2D(pool_size=(8, 8))(input_layer) 
    x = Dense(units=32, activation='relu')(x)
    x = Dense(units=3 * 32 * 32, activation='relu')(x) 
    x = keras.layers.Reshape((32, 32, 3))(x)
    x = x * input_layer 

    # Branch Path
    branch_x = input_layer

    # Combine Outputs
    x = Concatenate()([x, branch_x])

    x = Flatten()(x)
    x = Dense(units=128, activation='relu')(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer) 

    return model