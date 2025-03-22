import keras
from keras.layers import Input, Lambda, Conv2D, Add, Concatenate, Flatten, Dense
from tensorflow.keras import backend as K
from tensorflow.keras import layers

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def split_input(x):
        return tf.split(x, num_or_size_splits=3, axis=-1)

    split_input_layer = Lambda(split_input)(input_layer)

    group1 = split_input_layer[0]
    group2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_input_layer[1])
    group3 = split_input_layer[2]
    combined_group2_3 = Concatenate()([group2, group3])
    combined_group2_3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(combined_group2_3)
    combined_group1_2_3 = Concatenate()([group1, combined_group2_3])

    branch_path = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    main_path_output = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(combined_group1_2_3)
    combined_output = Add()([main_path_output, branch_path])

    bath_norm = layers.BatchNormalization()(combined_output)
    flatten_layer = Flatten()(bath_norm)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model