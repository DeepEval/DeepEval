import keras
from keras.layers import Input, GlobalAveragePooling2D, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Global Average Pooling
    pooling_layer = GlobalAveragePooling2D()(input_layer)

    # Fully connected layers
    dense_layer_1 = Dense(64, activation='relu')(pooling_layer)
    dense_layer_2 = Dense(128, activation='relu')(dense_layer_1)
    dense_layer_3 = Dense(32, activation='relu')(dense_layer_2)

    # Flatten the output
    flatten_layer = Flatten()(dense_layer_3)

    # Final output layer
    output_layer = Dense(10, activation='softmax')(flatten_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model