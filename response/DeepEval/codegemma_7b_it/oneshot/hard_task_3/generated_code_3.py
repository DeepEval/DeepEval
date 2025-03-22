import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Lambda

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    split = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)

    def block(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        drop = Dropout(0.25)(conv2)
        return drop

    main_pathway = Lambda(block)(split[0])
    main_pathway = Lambda(block)(main_pathway)
    main_pathway = Lambda(block)(main_pathway)
    main_pathway = Lambda(block)(main_pathway)

    branch_pathway = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split[1])
    branch_pathway = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_pathway)

    concat = Lambda(lambda x: tf.concat(x, axis=3))([main_pathway, branch_pathway])
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(concat)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model