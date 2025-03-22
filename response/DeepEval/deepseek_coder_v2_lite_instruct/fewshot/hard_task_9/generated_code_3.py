import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense, Concatenate

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path branches
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch2)
    branch3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch3)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch3)
    
    # Fusing branches through addition
    added_branches = Add()([branch1, branch2, branch3])
    
    # Adjusting output dimensions
    final_branch = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(added_branches)
    
    # Flatten the output
    flatten_layer = Flatten()(final_branch)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model