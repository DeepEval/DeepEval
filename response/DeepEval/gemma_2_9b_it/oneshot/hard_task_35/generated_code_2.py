import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Reshape, Concatenate, Flatten
from keras.applications.cifar10 import preprocess_input

def dl_model():  
    
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1
    x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    pool1 = GlobalAveragePooling2D()(x)
    fc1_1 = Dense(units=32, activation='relu')(pool1)
    fc2_1 = Dense(units=3, activation='relu')(fc1_1)  
    branch1_output = Reshape((32, 32, 3))(fc2_1)  
    branch1_output = keras.layers.Multiply()([x, branch1_output]) 

    # Branch 2
    y = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    y = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(y)
    pool2 = GlobalAveragePooling2D()(y)
    fc1_2 = Dense(units=32, activation='relu')(pool2)
    fc2_2 = Dense(units=3, activation='relu')(fc1_2)
    branch2_output = Reshape((32, 32, 3))(fc2_2)
    branch2_output = keras.layers.Multiply()([y, branch2_output])

    # Concatenate branches
    concatenated_output = Concatenate()([branch1_output, branch2_output])
    
    # Flatten and final layer
    flatten_layer = Flatten()(concatenated_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model