import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense

def dl_model():
    def repeat_block(input_tensor):
        x = input_tensor
        for _ in range(3):  # Repeat the block 3 times
            bn = BatchNormalization()(x)
            relu = ReLU()(bn)
            conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='linear')(relu)
            x = Concatenate(axis=-1)([x, conv])  # Concatenate the original input with the new features
        return x

    input_layer = Input(shape=(28, 28, 1))
    
    # First pathway
    path1_output = repeat_block(input_layer)
    
    # Second pathway
    path2_output = repeat_block(input_layer)
    
    # Concatenate outputs from both pathways
    merged_output = Concatenate(axis=-1)([path1_output, path2_output])
    
    # Classification layers
    flatten_layer = Flatten()(merged_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model