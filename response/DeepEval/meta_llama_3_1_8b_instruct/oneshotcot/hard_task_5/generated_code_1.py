import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Reshape, Permute, DepthwiseConv2D
from keras import backend as K
import tensorflow as tf

def dl_model():    

    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    block1 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1[0])
    conv1_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1[1])
    conv1_3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1[2])
    fused_features = Concatenate()([conv1_1, conv1_2, conv1_3])

    # Block 2
    shape = Lambda(lambda x: K.int_shape(x))(fused_features)
    height, width, channels = shape[1], shape[2], shape[3]
    block2 = Lambda(lambda x: tf.reshape(x, (-1, height, width, 3, channels//3)))(fused_features)
    block2 = Lambda(lambda x: tf.transpose(x, (0, 1, 2, 4, 3)))(block2)
    block2 = Lambda(lambda x: tf.reshape(x, (-1, height, width, channels)))(block2)

    # Block 3
    block3 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2)

    # Branch
    branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine main path and branch
    combined = Add()([block3, branch])

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(Flatten()(combined))

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model