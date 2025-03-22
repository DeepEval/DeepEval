import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, Dropout, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    def specialized_block(input_tensor):
        x = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        x = Dropout(0.3)(x)
        x = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = Dropout(0.3)(x)
        return x

    # Three parallel branches
    branch1 = specialized_block(input_layer)
    branch2 = specialized_block(input_layer)
    branch3 = specialized_block(input_layer)

    # Concatenate the outputs of the branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Fully connected layers
    flatten_layer = Flatten()(concatenated)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model