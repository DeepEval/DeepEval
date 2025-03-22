import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense

def dl_model():
    def block(input_tensor):
        bn = BatchNormalization()(input_tensor)
        relu = ReLU()(bn)
        conv = Conv2D(filters=input_tensor.shape[-1], kernel_size=(3, 3), strides=(1, 1), padding='same')(relu)
        output_tensor = Concatenate(axis=-1)([input_tensor, conv])
        return output_tensor

    input_layer = Input(shape=(28, 28, 1))
    
    # Pathway 1
    path1 = block(input_layer)
    path1 = block(path1)
    path1 = block(path1)
    
    # Pathway 2
    path2 = block(input_layer)
    path2 = block(path2)
    path2 = block(path2)
    
    # Merge both pathways
    merged = Concatenate(axis=-1)([path1, path2])
    
    # Classification layers
    flatten = Flatten()(merged)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model