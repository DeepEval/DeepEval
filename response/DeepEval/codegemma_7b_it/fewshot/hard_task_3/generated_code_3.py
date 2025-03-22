import keras
import tensorflow as tf
from keras.layers import Input, Lambda, Flatten, Concatenate, Conv2D, Dropout, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def group(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        drop = Dropout(rate=0.2)(conv2)
        return drop

    groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)
    group1_output = group(groups[0])
    group2_output = group(groups[1])
    group3_output = group(groups[2])

    main_pathway_output = Concatenate()([group1_output, group2_output, group3_output])

    branch_pathway_output = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    combined_output = tf.add([main_pathway_output, branch_pathway_output])

    flatten = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

model = dl_model()