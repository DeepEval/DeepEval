import keras
from keras.layers import Input, Conv2D, Dropout, Concatenate, Flatten, Dense

def dl_model():  
    
    input_layer = Input(shape=(28, 28, 1))

    # Main Path
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    x = Dropout(0.2)(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x) 

    # Branch Path
    branch_x = input_layer
    branch_x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch_x) 

    # Concatenate outputs
    x = Concatenate()([x, branch_x]) 

    # Flatten and output layer
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer) 
    return model