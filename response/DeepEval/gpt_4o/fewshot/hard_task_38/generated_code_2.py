import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    def repeated_block(input_tensor):
        # Loop to create the block structure three times
        x = input_tensor
        for _ in range(3):
            bn = BatchNormalization()(x)
            relu = ReLU()(bn)
            conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu)
            x = Concatenate(axis=-1)([x, conv])
        return x
    
    # First Pathway
    path1_output = repeated_block(input_tensor=input_layer)
    
    # Second Pathway
    path2_output = repeated_block(input_tensor=input_layer)

    # Merge outputs of both pathways through concatenation
    merged_output = Concatenate(axis=-1)([path1_output, path2_output])
    
    # Classification using fully connected layers
    flatten_layer = Flatten()(merged_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model