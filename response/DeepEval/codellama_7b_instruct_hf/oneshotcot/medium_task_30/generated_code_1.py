import keras
from keras.layers import Input, AveragePooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Three average pooling layers with pooling windows and strides of 1x1, 2x2, and 4x4
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(pool1)
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(pool2)

    # Flatten the outputs of the pooling layers
    flatten1 = Flatten()(pool1)
    flatten2 = Flatten()(pool2)
    flatten3 = Flatten()(pool3)

    # Concatenate the flattened outputs
    concat = keras.layers.concatenate([flatten1, flatten2, flatten3])

    # Two fully connected layers to generate the final classification results
    dense1 = Dense(units=128, activation='relu')(concat)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model