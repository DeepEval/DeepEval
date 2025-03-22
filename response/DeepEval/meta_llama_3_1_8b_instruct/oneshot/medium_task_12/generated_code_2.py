import keras
from keras.layers import Input, Conv2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model
from keras.regularizers import l2

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', 
                       kernel_regularizer=l2(0.01))(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', 
                       kernel_regularizer=l2(0.01))(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', 
                       kernel_regularizer=l2(0.01))(conv2)
        batch_norm = BatchNormalization()(conv3)
        output_tensor = Concatenate()([conv1, conv2, conv3, batch_norm])

        return output_tensor

    block1_output = block(input_layer)
    block2_output = block(block1_output)
    block3_output = block(block2_output)

    bath_norm = BatchNormalization()(block3_output)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model