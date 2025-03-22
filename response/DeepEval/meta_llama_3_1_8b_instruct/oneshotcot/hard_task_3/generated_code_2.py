import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Dropout, Dense, Add
from tensorflow.keras.layers import AveragePooling2D
import tensorflow as tf

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    def split_input(input_tensor):
        return tf.split(input_tensor, num_or_size_splits=3, axis=-1)

    split_layer = Lambda(split_input)(input_layer)

    group1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
    group1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group1)
    group1 = Dropout(0.2)(group1)
    group1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(group1)

    group2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[1])
    group2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group2)
    group2 = Dropout(0.2)(group2)
    group2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(group2)

    group3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[2])
    group3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group3)
    group3 = Dropout(0.2)(group3)
    group3 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(group3)

    main_pathway = Concatenate()([group1, group2, group3])

    branch_pathway = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
    branch_pathway = Add()([main_pathway, branch_pathway])

    flatten_layer = Flatten()(branch_pathway)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model