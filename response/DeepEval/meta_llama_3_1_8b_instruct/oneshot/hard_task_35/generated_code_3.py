import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape, GlobalAveragePooling2D, Multiply
from keras import regularizers
from keras import backend as K

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1 = BatchNormalization()(branch1)
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch1)

    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = BatchNormalization()(branch2)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch2)

    def block(input_tensor):

        avg_pool = GlobalAveragePooling2D()(input_tensor)
        weights = Dense(units=32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(avg_pool)
        weights = Reshape((1, 1, 32))(weights)
        weighted_output = Multiply()([input_tensor, weights])
        output_tensor = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(weighted_output)
        return output_tensor

    branch1_output = block(branch1)
    branch2_output = block(branch2)
    output = Concatenate()([branch1_output, branch2_output])
    output = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(output)
    output = Flatten()(output)
    output = Dense(units=128, activation='relu')(output)
    output = Dense(units=10, activation='softmax')(output)

    model = keras.Model(inputs=input_layer, outputs=output)

    return model