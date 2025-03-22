import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Reshape, multiply, Concatenate, Flatten

def dl_model():     
    input_tensor = Input(shape=(32, 32, 3))

    # Branch 1
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    branch1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
    branch1_pool = GlobalAveragePooling2D()(branch1)
    branch1_dense1 = Dense(units=128, activation='relu')(branch1_pool)
    branch1_dense2 = Dense(units=3, activation='relu')(branch1_dense1)  # Adjust output dimension as needed
    branch1_reshape = Reshape((32, 32, 3))(branch1_dense2) 
    branch1_output = multiply([branch1, branch1_reshape])

    # Branch 2
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    branch2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2_pool = GlobalAveragePooling2D()(branch2)
    branch2_dense1 = Dense(units=128, activation='relu')(branch2_pool)
    branch2_dense2 = Dense(units=3, activation='relu')(branch2_dense1)  # Adjust output dimension as needed
    branch2_reshape = Reshape((32, 32, 3))(branch2_dense2) 
    branch2_output = multiply([branch2, branch2_reshape])

    # Concatenate branches
    combined_output = Concatenate()([branch1_output, branch2_output])

    # Flatten and final dense layer
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  

    model = keras.Model(inputs=input_tensor, outputs=output_layer)
    return model