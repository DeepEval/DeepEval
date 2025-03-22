import keras
from keras.layers import Input, SeparableConv2D, Conv2D, Dropout, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    def branch(input_tensor):
        sep_conv = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        dropout1 = Dropout(0.2)(sep_conv)
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(dropout1)
        dropout2 = Dropout(0.2)(conv1x1)
        return dropout2
    
    branch1 = branch(input_layer)
    branch2 = branch(input_layer)
    branch3 = branch(input_layer)
    
    concatenated_branches = Concatenate()([branch1, branch2, branch3])
    
    flatten_layer = Flatten()(concatenated_branches)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model