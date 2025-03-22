import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Dense, Flatten

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    def processing_block(input_tensor):
        for _ in range(3):  # Repeat the block 3 times
            x = BatchNormalization()(input_tensor)
            x = ReLU()(x)
            x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
            input_tensor = Concatenate()([input_tensor, x])  # Concatenate original input with new features
        return input_tensor

    # First processing pathway
    path1_output = processing_block(input_layer)

    # Second processing pathway
    path2_output = processing_block(input_layer)

    # Concatenate outputs from both pathways
    merged_output = Concatenate()([path1_output, path2_output])

    # Classification layers
    flatten = Flatten()(merged_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model