import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        # Batch Normalization and ReLU Activation
        batch_norm = BatchNormalization()(input_tensor)
        relu = ReLU()(batch_norm)

        # 3x3 Convolution Layer
        conv_layer = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu)
        
        # Concatenate original input with the features from conv_layer
        concatenated = Concatenate(axis=-1)([input_tensor, conv_layer])

        return concatenated

    # First pathway
    path1 = input_layer
    for _ in range(3):
        path1 = block(path1)

    # Second pathway
    path2 = input_layer
    for _ in range(3):
        path2 = block(path2)

    # Merge both pathways
    merged_output = Concatenate()([path1, path2])

    # Classification layers
    flatten_layer = Flatten()(merged_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model