import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense, Concatenate

def dl_model(): 
    
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        x1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
        x2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x1)
        x3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x2)

        parallel_path = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
        
        output_tensor = Add()([x3, parallel_path])
        return output_tensor
    
    branch1 = block(input_layer)
    branch2 = block(input_layer)
    concatenated_output = Concatenate()([branch1, branch2])
    
    flatten_layer = Flatten()(concatenated_output)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense_layer)

    return model