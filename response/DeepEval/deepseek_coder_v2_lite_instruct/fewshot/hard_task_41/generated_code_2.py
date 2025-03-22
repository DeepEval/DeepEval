import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Dropout
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block_1(input_tensor):
        avgpool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(avgpool1)
        avgpool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(avgpool2)
        avgpool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(avgpool3)
        concatenated = Concatenate()([flatten1, flatten2, flatten3])
        dropout = Dropout(0.5)(concatenated)  # Dropout for regularization
        return dropout

    block1_output = block_1(input_tensor=input_layer)
    dense = Dense(units=64, activation='relu')(block1_output)
    reshaped = Reshape(target_shape=(4, 4, 4))(dense)

    def block_2(input_tensor):
        # Branch 1: 1x1 convolution
        conv1 = Dense(units=32, activation='relu')(input_tensor)
        # Branch 2: <1x1 convolution
        conv2 = Dense(units=32, activation='relu')(input_tensor)
        # Branch 3: 3x3 convolution
        conv3 = Dense(units=32, activation='relu')(input_tensor)
        # Branch 4: <1x1 convolution, 3x3 convolution, 3x3 convolution>
        conv4 = Dense(units=32, activation='relu')(input_tensor)
        concatenated = Concatenate()([conv1, conv2, conv3, conv4])
        return concatenated

    block2_output = block_2(input_tensor=reshaped)
    flattened = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flattened)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model