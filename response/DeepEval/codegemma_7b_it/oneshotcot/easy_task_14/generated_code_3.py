import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D

def dl_model():

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Global average pooling layer
    global_avg_pool = GlobalAveragePooling2D()(input_layer)

    # Fully connected layers to generate weights
    dense1 = Dense(units=32, activation='relu')(global_avg_pool)
    dense2 = Dense(units=32, activation='relu')(dense1)

    # Reshape weights to align with input shape
    weights = Dense(units=32, activation='softmax')(dense2)
    weights = Reshape((32,))(weights)

    # Multiply weights with input feature map
    weighted_input = Multiply()([input_layer, weights])

    # Flatten and fully connected layer for output
    flatten_layer = Flatten()(weighted_input)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Model definition
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model