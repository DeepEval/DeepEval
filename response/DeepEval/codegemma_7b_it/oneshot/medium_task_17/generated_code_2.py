import keras
from keras import layers
from keras.models import Model

def dl_model():
    input_layer = layers.Input(shape=(32, 32, 3))
    input_shape = keras.backend.int_shape(input_layer)

    # Reshape and group the input tensor
    input_tensor = layers.Reshape((input_shape[1], input_shape[2], 3, input_shape[3]//3))(input_layer)
    input_tensor = layers.Permute((0, 1, 3, 2))(input_tensor)

    # Reshape back to the original input shape
    input_tensor = layers.Reshape((input_shape[1], input_shape[2], input_shape[3]))(input_tensor)

    # Fully connected layer for classification
    output_layer = layers.Dense(units=10, activation='softmax')(input_tensor)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model