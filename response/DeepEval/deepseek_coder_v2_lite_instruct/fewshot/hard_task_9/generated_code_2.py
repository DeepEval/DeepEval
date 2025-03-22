import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path branches
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch2)
    branch3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch3)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch3)
    
    # Adjusting the output dimensions of the branches
    branch1 = Conv2D(filters=3, kernel_size=(1, 1), activation='relu')(branch1)
    branch2 = Conv2D(filters=3, kernel_size=(1, 1), activation='relu')(branch2)
    branch3 = Conv2D(filters=3, kernel_size=(1, 1), activation='relu')(branch3)
    
    # Fusing the branches through addition
    added = Add()([branch1, branch2, branch3])
    
    # Flattening the fused output
    flattened = Flatten()(added)
    
    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Constructing the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model