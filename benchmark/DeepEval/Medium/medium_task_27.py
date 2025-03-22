from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Softmax, Multiply, Add

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
        
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',activation='relu')(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same',activation='relu')(input_layer)
    add_brancher = Add()([branch1, branch2])
    
    global_pool = GlobalAveragePooling2D()(add_brancher)
    dense = Dense(units=64, activation='relu')(global_pool)
    dense = Dense(units=64, activation='relu')(dense)
    
    attention_weights = Softmax()(dense)
    weighted_branch1 = Multiply()([branch1, attention_weights])
    weighted_branch2 = Multiply()([branch2, attention_weights])
    
    added = Add()([weighted_branch1, weighted_branch2])
    output_layer = Dense(units=10, activation='softmax')(added)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
