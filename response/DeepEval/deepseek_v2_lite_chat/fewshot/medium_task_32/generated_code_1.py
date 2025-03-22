import keras
from keras.layers import Input, Lambda, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense, Reshape

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        split1 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)
        conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split1[0])
        conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split1[1])
        conv1_3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split1[2])
        output_tensor = Concatenate()([conv1_1, conv1_2, conv1_3])
        return output_tensor

    def block_2(input_tensor):
        conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        output_tensor = Flatten()(conv2)
        dense = Dense(units=64, activation='relu')(output_tensor)
        reshaped = Reshape(target_shape=(64,))((dense))
        output_tensor = Dense(units=10, activation='softmax')(reshaped)
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    model = block_2(input_tensor=block1_output)

    return model