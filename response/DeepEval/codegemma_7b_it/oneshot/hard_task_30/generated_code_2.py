import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # First block
    def block1(input_tensor):
        main_path = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        main_path = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
        main_path = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path)
        branch_path = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        output_tensor = keras.layers.add([main_path, branch_path])
        return output_tensor

    # Second block
    def block2(input_tensor):
        group1 = Lambda(lambda x: keras.backend.tf.split(x, 3, axis=3))(input_tensor)
        group1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1)
        group2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group1)
        group3 = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(group1)
        output_tensor = keras.layers.concatenate([group1, group2, group3])
        return output_tensor

    # Model definition
    block1_output = block1(input_layer)
    block2_output = block2(block1_output)
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

model = dl_model()