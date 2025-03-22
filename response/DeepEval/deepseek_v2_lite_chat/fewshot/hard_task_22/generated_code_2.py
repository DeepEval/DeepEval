import keras
from keras.layers import Input, Lambda, Conv2D, Add, Flatten, Dense, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def main_path(input_tensor):
        split1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split1[0])
        conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split1[1])
        conv1_3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split1[2])
        concatenated = Concatenate(axis=-1)([conv1_1, conv1_2, conv1_3])

        return concatenated

    def branch_path(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv1

    main_output = main_path(input_layer)
    branch_output = branch_path(input_tensor=input_layer)
    fused_output = Add()([main_output, branch_output])
    flatten = Flatten()(fused_output)

    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

model = dl_model()
model.summary()