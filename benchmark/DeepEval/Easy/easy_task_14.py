from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Reshape, Multiply,Flatten,Dense

def dl_model():

    input_layer = Input(shape=(32,32,3))

    squeeze = GlobalAveragePooling2D()(input_layer)

    input_channels = input_layer.shape[-1]
    excitation_1 = Dense(units=input_channels // 3, activation='relu')(squeeze) 
    excitation_2 = Dense(units=input_channels, activation='sigmoid')(excitation_1)
    reshaped = Reshape(target_shape=(1, 1, input_channels))(excitation_2)  
    scaled = Multiply()([input_layer, reshaped]) 

    flatten = Flatten()(scaled)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

