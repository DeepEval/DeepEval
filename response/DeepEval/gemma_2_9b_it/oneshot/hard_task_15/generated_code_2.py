import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Concatenate, Flatten, Dense, Reshape

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    x = GlobalAveragePooling2D()(input_layer)
    x = Dense(units=32, activation='relu')(x)
    x = Dense(units=32, activation='relu')(x)
    x = Reshape((32, 32, 3))(x) 
    x = x * input_layer  

    # Branch Path
    branch_x = input_layer

    # Combine Outputs
    output = Concatenate()([x, branch_x])
    output = Dense(units=128, activation='relu')(output)
    output = Dense(units=10, activation='softmax')(output)

    model = keras.Model(inputs=input_layer, outputs=output)

    return model