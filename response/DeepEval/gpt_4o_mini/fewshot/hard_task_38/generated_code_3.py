import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        # Batch normalization
        norm = BatchNormalization()(input_tensor)
        # ReLU activation
        relu = ReLU()(norm)
        # Convolutional layer
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu)
        # Concatenate the original input with the features
        output_tensor = Concatenate()([input_tensor, conv])
        return output_tensor
    
    # First pathway
    path1 = input_layer
    for _ in range(3):  # Repeat the block 3 times
        path1 = block(path1)

    # Second pathway
    path2 = input_layer
    for _ in range(3):  # Repeat the block 3 times
        path2 = block(path2)

    # Concatenate the outputs of both pathways
    merged_output = Concatenate()([path1, path2])

    # Flatten and apply two fully connected layers for classification
    flatten_layer = Flatten()(merged_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model