import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, Dropout, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def branch(input_tensor):
        x = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        x = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(x)
        x = Dropout(rate=0.3)(x)
        return x

    # Create three branches
    branch1_output = branch(input_layer)
    branch2_output = branch(input_layer)
    branch3_output = branch(input_layer)

    # Concatenate the outputs from the three branches
    concatenated = Concatenate()([branch1_output, branch2_output, branch3_output])

    # Flatten the concatenated output
    flatten_layer = Flatten()(concatenated)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dropout1 = Dropout(rate=0.5)(dense1)
    dense2 = Dense(units=64, activation='relu')(dropout1)
    dropout2 = Dropout(rate=0.5)(dense2)

    # Output layer with softmax activation for classification
    output_layer = Dense(units=10, activation='softmax')(dropout2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model