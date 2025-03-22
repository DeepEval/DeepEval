import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate, Flatten

def dl_model(): 
    
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(branch1)
    branch1_pool = GlobalAveragePooling2D()(branch1)
    branch1_fc1 = Dense(units=128, activation='relu')(branch1_pool)
    branch1_fc2 = Dense(units=3, activation='relu')(branch1_fc1)  # Output channels match input channel
    branch1_reshaped = Reshape((32, 32, 3))(branch1_fc2) 
    branch1_output = Multiply()([branch1, branch1_reshaped])

    # Branch 2
    branch2 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    branch2 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(branch2)
    branch2_pool = GlobalAveragePooling2D()(branch2)
    branch2_fc1 = Dense(units=128, activation='relu')(branch2_pool)
    branch2_fc2 = Dense(units=3, activation='relu')(branch2_fc1)
    branch2_reshaped = Reshape((32, 32, 3))(branch2_fc2)
    branch2_output = Multiply()([branch2, branch2_reshaped])

    # Concatenate outputs
    concat_output = Concatenate()([branch1_output, branch2_output])

    # Flatten and final classification
    flatten = Flatten()(concat_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)  

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model