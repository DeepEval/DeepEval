import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):

        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        batch_norm1 = BatchNormalization()(conv1)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(batch_norm1)
        batch_norm2 = BatchNormalization()(conv2)
        output_tensor = Concatenate()([conv1, conv2])

        return output_tensor

    pathway1 = block(input_tensor)
    pathway2 = block(input_tensor)

    merged_output = Concatenate()([pathway1, pathway2])

    flatten_layer = Flatten()(merged_output)
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model