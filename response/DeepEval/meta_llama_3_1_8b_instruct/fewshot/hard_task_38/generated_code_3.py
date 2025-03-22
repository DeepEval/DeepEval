import keras
from keras.layers import Input, Conv2D, BatchNormalization, Concatenate, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))
    
    def block(input_tensor):
        bn = BatchNormalization()(input_tensor)
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(bn)
        output_tensor = Concatenate()([input_tensor, conv])
        return output_tensor
    
    pathway_1 = input_layer
    for _ in range(3):
        pathway_1 = block(pathway_1)
    
    pathway_2 = input_layer
    for _ in range(3):
        pathway_2 = block(pathway_2)
    
    concat_pathways = Concatenate()([pathway_1, pathway_2])
    flatten_layer = Flatten()(concat_pathways)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model