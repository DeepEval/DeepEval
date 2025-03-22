import keras
from keras.layers import Input, Conv2D, Concatenate, BatchNormalization, ReLU, Flatten, Dense

def dl_model():
    def repeated_block(input_tensor):
        for _ in range(3):  # Repeated 3 times
            x = BatchNormalization()(input_tensor)
            x = ReLU()(x)
            x = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(x)
            input_tensor = Concatenate()([input_tensor, x])  # Concatenate along the channel dimension
        return input_tensor
    
    input_layer = Input(shape=(28, 28, 1))
    
    # Pathway 1
    path1 = repeated_block(input_layer)
    
    # Pathway 2
    path2 = repeated_block(input_layer)
    
    # Concatenate outputs from both pathways
    merged = Concatenate()([path1, path2])
    
    # Classification part
    flatten_layer = Flatten()(merged)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model