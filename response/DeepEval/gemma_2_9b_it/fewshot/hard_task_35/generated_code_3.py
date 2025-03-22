import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Concatenate, Flatten

def dl_model():     
    input_layer = Input(shape=(32, 32, 3))

    def block(input_tensor):
        x = GlobalAveragePooling2D()(input_tensor)
        x = Dense(units=256, activation='relu')(x)
        x = Dense(units=3, activation='relu')(x)
        x = Reshape(target_shape=(32, 32, 3))(x)
        x = input_tensor * x 
        return x

    branch1_output = block(input_layer)
    branch2_output = block(input_layer)
    
    merged_tensor = Concatenate()([branch1_output, branch2_output])
    
    flatten_layer = Flatten()(merged_tensor)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model