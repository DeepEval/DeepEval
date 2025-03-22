import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)

    # Path 1: 1x1 separable convolution
    path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split[0])
    path1 = BatchNormalization()(path1)

    # Path 2: 3x3 separable convolution
    path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split[1])
    path2 = BatchNormalization()(path2)

    # Path 3: 5x5 separable convolution
    path3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split[2])
    path3 = BatchNormalization()(path3)

    # Concatenate outputs of Block 1
    block1_output = Concatenate()([path1, path2, path3])

    # Block 2
    # Path 1: 1x1 convolution
    path1_b2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(block1_output)

    # Path 2: 3x3 average pooling followed by 1x1 convolution
    path2_b2 = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(block1_output)
    path2_b2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(path2_b2)

    # Path 3: 1x1 convolution split into two paths
    path3_b2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(block1_output)
    path3_a = Conv2D(filters=32, kernel_size=(1, 3), padding='same', activation='relu')(path3_b2)
    path3_b = Conv2D(filters=32, kernel_size=(3, 1), padding='same', activation='relu')(path3_b2)
    path3_b2_output = Concatenate()([path3_a, path3_b])

    # Path 4: 1x1 convolution followed by 3x3 convolution and split
    path4_b2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(block1_output)
    path4_b2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(path4_b2)
    path4_a = Conv2D(filters=32, kernel_size=(1, 3), padding='same', activation='relu')(path4_b2)
    path4_b = Conv2D(filters=32, kernel_size=(3, 1), padding='same', activation='relu')(path4_b2)
    path4_b2_output = Concatenate()([path4_a, path4_b])

    # Concatenate outputs of Block 2
    block2_output = Concatenate()([path1_b2, path2_b2, path3_b2_output, path4_b2_output])

    # Flatten and Fully Connected Layer
    flatten_layer = Flatten()(block2_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model