import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Dropout, Flatten, Concatenate, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    def main_path(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
        drop1 = Dropout(0.5)(conv2)
        
        branch_input = conv2
        branch_output = branch_input
        
        # Flattening and adding branch output to main path output
        concat = Concatenate()([drop1, branch_output])
        flatten = Flatten()(concat)
        dense = Dense(units=128, activation='relu')(flatten)
        
        return dense
    
    branch_input = Input(shape=(32, 32, 3))
    branch_output = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(branch_input)
    
    # Dropout layer
    drop_branch = Dropout(0.5)(branch_output)
    
    # Flattening and fully connected layers
    flatten_branch = Flatten()(drop_branch)
    dense_branch = Dense(units=128, activation='relu')(flatten_branch)
    
    # Output layer
    output = Dense(units=10, activation='softmax')(dense_branch)
    
    # Model construction
    model = keras.Model(inputs=[input_layer, branch_input], outputs=output)
    
    return model