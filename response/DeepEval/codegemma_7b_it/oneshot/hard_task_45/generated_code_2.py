import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Lambda, DepthwiseConv2D, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    def split_and_conv(x):
        split_outputs = tf.split(x, 3, axis=-1)
        conv_outputs = []
        for i in range(3):
            conv_outputs.append(DepthwiseConv2D(kernel_size=(i + 1, i + 1), padding='same', activation='relu')(split_outputs[i]))
        return tf.concat(conv_outputs, axis=-1)

    split_conv = Lambda(split_and_conv)(input_layer)

    # Second block
    def branch(x):
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
        if x.shape[2] == 32:
            conv3x3_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
            conv3x3_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3x3_1)
            maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)
            conv1x1_maxpool = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(maxpool)
            return tf.concat([conv1x1, conv3x3_2, conv1x1_maxpool], axis=-1)
        else:
            conv3x3_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
            conv3x3_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3x3_1)
            return tf.concat([conv1x1, conv3x3_2], axis=-1)

    branches = [branch(split_conv)]
    for i in range(4):
        branches.append(branch(branches[-1]))

    concat_branches = tf.concat(branches, axis=-1)

    # Output layer
    flatten_layer = Flatten()(concat_branches)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model