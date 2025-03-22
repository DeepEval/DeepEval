import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Define the block
    def block(input_tensor):
      
        pooled = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=input_tensor.shape[-1], activation='relu')(pooled)
        dense2 = Dense(units=input_tensor.shape[-1], activation='relu')(dense1)
        weights = Reshape((input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[-1]))(dense2)
        output_tensor = keras.layers.multiply([input_tensor, weights])
        return output_tensor

    # Create the two branches
    branch1 = block(input_layer)
    branch2 = block(input_layer)

    # Concatenate the outputs of the branches
    merged = Concatenate()([branch1, branch2])

    # Flatten and add the final fully connected layer
    flatten_layer = Flatten()(merged)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model