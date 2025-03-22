import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    def block(input_tensor):
        # Apply batch normalization and ReLU activation
        x = BatchNormalization()(input_tensor)
        x = ReLU()(x)
        # Apply 3x3 convolution
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        # Concatenate the original input with the new features
        output_tensor = Concatenate()([input_tensor, x])
        return output_tensor

    # First pathway with three repeated blocks
    pathway1 = input_layer
    for _ in range(3):
        pathway1 = block(pathway1)
    
    # Second pathway with three repeated blocks
    pathway2 = input_layer
    for _ in range(3):
        pathway2 = block(pathway2)

    # Concatenate the outputs from both pathways
    merged_output = Concatenate()([pathway1, pathway2])

    # Flatten the merged output
    flatten_layer = Flatten()(merged_output)
    
    # Two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model