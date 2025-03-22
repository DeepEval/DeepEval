import keras
import tensorflow as tf
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, Dropout, Flatten, Dense, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        dropout = Dropout(0.2)(conv3)
        output_tensor = Concatenate()([conv1, conv2, dropout])
        return output_tensor

    def block_2(input_tensor):
        branches = [
            [input_tensor, Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')],
            [input_tensor, Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'),
             Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')],
            [input_tensor, Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'),
             Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')],
            [input_tensor, MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same'),
             Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')]
        ]
        outputs = []
        for branch in branches:
            x = branch[0]
            for layer in branch[1:]:
                x = layer(x)
            outputs.append(x)
        output_tensor = Concatenate()(outputs)
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)

    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model