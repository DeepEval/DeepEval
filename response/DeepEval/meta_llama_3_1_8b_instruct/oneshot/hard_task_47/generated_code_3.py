import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, AveragePooling2D
from keras import backend as K
import tensorflow as tf

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    split_layer = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)

    # Group 1
    conv1_1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same')(split_layer[0])
    conv1_1 = Conv2D(64, (1, 1), padding='same')(conv1_1)
    conv1_1 = BatchNormalization()(conv1_1)
    conv1_1 = tf.nn.relu(conv1_1)

    # Group 2
    conv2_1 = DepthwiseConv2D(kernel_size=(3, 3), padding='same')(split_layer[1])
    conv2_1 = Conv2D(64, (1, 1), padding='same')(conv2_1)
    conv2_1 = BatchNormalization()(conv2_1)
    conv2_1 = tf.nn.relu(conv2_1)

    # Group 3
    conv3_1 = DepthwiseConv2D(kernel_size=(5, 5), padding='same')(split_layer[2])
    conv3_1 = Conv2D(64, (1, 1), padding='same')(conv3_1)
    conv3_1 = BatchNormalization()(conv3_1)
    conv3_1 = tf.nn.relu(conv3_1)

    concatenated_layer = Concatenate()([conv1_1, conv2_1, conv3_1])

    # Second Block
    conv4_1 = Conv2D(64, (1, 1), padding='same')(concatenated_layer)
    conv4_1 = BatchNormalization()(conv4_1)
    conv4_1 = tf.nn.relu(conv4_1)

    conv4_2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same')(concatenated_layer)
    conv4_2 = Conv2D(64, (1, 1), padding='same')(conv4_2)
    conv4_2 = BatchNormalization()(conv4_2)
    conv4_2 = tf.nn.relu(conv4_2)

    conv4_3 = DepthwiseConv2D(kernel_size=(1, 7), padding='same')(concatenated_layer)
    conv4_3 = Conv2D(64, (1, 1), padding='same')(conv4_3)
    conv4_3 = BatchNormalization()(conv4_3)
    conv4_3 = tf.nn.relu(conv4_3)

    conv4_4 = DepthwiseConv2D(kernel_size=(7, 1), padding='same')(concatenated_layer)
    conv4_4 = Conv2D(64, (1, 1), padding='same')(conv4_4)
    conv4_4 = BatchNormalization()(conv4_4)
    conv4_4 = tf.nn.relu(conv4_4)

    conv4_5 = DepthwiseConv2D(kernel_size=(3, 3), padding='same')(concatenated_layer)
    conv4_5 = Conv2D(64, (1, 1), padding='same')(conv4_5)
    conv4_5 = BatchNormalization()(conv4_5)
    conv4_5 = tf.nn.relu(conv4_5)

    avg_pool = AveragePooling2D(pool_size=(8, 8))(concatenated_layer)
    avg_pool = Flatten()(avg_pool)

    concatenated_layer2 = Concatenate()([conv4_1, conv4_2, conv4_3, conv4_4, conv4_5, avg_pool])

    dense1 = Dense(units=128, activation='relu')(concatenated_layer2)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model