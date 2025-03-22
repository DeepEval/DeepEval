import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense

def dl_model():
    def block(input_tensor):
        # Apply batch normalization
        x = BatchNormalization()(input_tensor)
        # Apply ReLU activation
        x = ReLU()(x)
        # Apply a 3x3 convolutional layer
        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x)
        # Concatenate the original input with the new features
        output_tensor = Concatenate(axis=-1)([input_tensor, x])
        return output_tensor
    
    input_layer = Input(shape=(28, 28, 1))

    # Pathway 1: Apply the block structure three times
    pathway1 = block(input_layer)
    pathway1 = block(pathway1)
    pathway1 = block(pathway1)

    # Pathway 2: Apply the block structure three times
    pathway2 = block(input_layer)
    pathway2 = block(pathway2)
    pathway2 = block(pathway2)

    # Concatenate the outputs from both pathways
    merged = Concatenate(axis=-1)([pathway1, pathway2])

    # Flatten the result
    flatten_layer = Flatten()(merged)
    # Fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    # Final classification layer
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model