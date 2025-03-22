import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def repeated_block(input_tensor):
        # Define a single block with BatchNormalization, ReLU, and Conv2D
        x = BatchNormalization()(input_tensor)
        x = ReLU()(x)
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        # Concatenate the input with the output of the conv layer
        output = Concatenate()([input_tensor, x])
        return output

    # First pathway
    path1 = repeated_block(input_layer)
    path1 = repeated_block(path1)
    path1 = repeated_block(path1)

    # Second pathway
    path2 = repeated_block(input_layer)
    path2 = repeated_block(path2)
    path2 = repeated_block(path2)

    # Concatenate the outputs of both pathways
    merged = Concatenate()([path1, path2])

    # Flatten the merged output
    flatten_layer = Flatten()(merged)

    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model