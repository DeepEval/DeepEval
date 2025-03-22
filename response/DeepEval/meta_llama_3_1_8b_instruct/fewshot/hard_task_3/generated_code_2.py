import keras
from keras.layers import Input, Lambda, Conv2D, Dropout, Add, Flatten, Dense
import tensorflow as tf

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    def block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        dropout = Dropout(0.2)(conv2)
        return dropout

    inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)
    group1 = block(inputs_groups[0])
    group2 = block(inputs_groups[1])
    group3 = block(inputs_groups[2])
    main_path = Concatenate()([group1, group2, group3])

    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    adding_layer = Add()([main_path, branch_path])

    flatten_layer = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model