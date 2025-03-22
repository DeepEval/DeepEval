import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dense, Concatenate
from keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Define the first block
    block1_output = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    block1_output = Flatten()(block1_output)
    block1_output = Dense(units=64, activation='relu')(block1_output)
    block1_output = Reshape(target_shape=(4, 4, 4))(block1_output)

    # Define the second block
    block2_output = Concatenate()([
        MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(block1_output),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(block1_output),
        MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(block1_output)
    ])
    block2_output = Flatten()(block2_output)
    block2_output = Dense(units=128, activation='relu')(block2_output)
    block2_output = Dense(units=64, activation='relu')(block2_output)
    block2_output = Dense(units=10, activation='softmax')(block2_output)

    # Define the model
    model = Model(inputs=input_layer, outputs=block2_output)

    return model