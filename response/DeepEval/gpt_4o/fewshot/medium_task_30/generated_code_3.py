import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First Average Pooling Layer with 1x1 window
    avgpool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    flatten1 = Flatten()(avgpool1)

    # Second Average Pooling Layer with 2x2 window
    avgpool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    flatten2 = Flatten()(avgpool2)

    # Third Average Pooling Layer with 4x4 window
    avgpool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)
    flatten3 = Flatten()(avgpool3)

    # Concatenation of the flattened outputs from the pooling layers
    concatenated_features = Concatenate()([flatten1, flatten2, flatten3])

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(concatenated_features)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model