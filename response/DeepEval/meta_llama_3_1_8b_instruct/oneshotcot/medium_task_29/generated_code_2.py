import keras
from keras.layers import Input, MaxPooling2D, Concatenate, Flatten, Dense
from keras.models import Model
from keras.layers import Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First max pooling layer with window size 1x1
    max_pooling_1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_layer)

    # Reshape the output to prepare it for concatenation
    reshape_1 = Reshape((32*32, 3))(max_pooling_1)

    # Second max pooling layer with window size 2x2
    max_pooling_2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    reshape_2 = Reshape((16*16*3,))(max_pooling_2)

    # Third max pooling layer with window size 4x4
    max_pooling_3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(input_layer)
    reshape_3 = Reshape((8*8*3,))(max_pooling_3)

    # Concatenate the outputs from the three pooling layers
    output_tensor = Concatenate()([reshape_1, reshape_2, reshape_3])

    # First fully connected layer
    dense1 = Dense(units=128, activation='relu')(output_tensor)
    # Second fully connected layer
    dense2 = Dense(units=64, activation='relu')(dense1)
    # Output layer with softmax activation for classification
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model