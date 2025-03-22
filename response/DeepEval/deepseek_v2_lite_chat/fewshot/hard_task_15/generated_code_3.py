import keras
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Dense, Flatten

def dl_model():

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    x = GlobalAveragePooling2D()(input_layer)
    x = Dense(units=512, activation='relu')(x)
    x = Dense(units=256, activation='relu')(x)
    
    # Branch path
    branch_input = input_layer

    # Add the outputs of the main and branch paths
    combined_output = keras.layers.Add()([x, branch_input])

    # Final fully connected layers
    x = Dense(units=10, activation='softmax')(combined_output)

    # Model
    model = Model(inputs=input_layer, outputs=x)

    return model