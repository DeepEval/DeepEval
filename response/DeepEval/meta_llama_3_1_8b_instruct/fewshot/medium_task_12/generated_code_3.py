import keras
from keras.layers import Input, Conv2D, BatchNormalization, Concatenate, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        batch_norm1 = BatchNormalization()(conv1)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(batch_norm1)
        batch_norm2 = BatchNormalization()(conv2)
        conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(batch_norm2)
        batch_norm3 = BatchNormalization()(conv3)
        output_tensor = Concatenate()([input_tensor, batch_norm1, batch_norm2, batch_norm3])
        return output_tensor

    block1_output = block(input_layer)
    block2_output = block(block1_output)
    block3_output = block(block2_output)

    flatten_layer = Flatten()(block3_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model