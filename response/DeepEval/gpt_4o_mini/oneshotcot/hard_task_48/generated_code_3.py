import keras
from keras.layers import Input, SeparableConv2D, BatchNormalization, Concatenate, Flatten, Dense, Lambda
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Path 1: 1x1 Separable Convolution
    path1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_inputs[0])
    path1 = BatchNormalization()(path1)

    # Path 2: 3x3 Separable Convolution
    path2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_inputs[1])
    path2 = BatchNormalization()(path2)

    # Path 3: 5x5 Separable Convolution
    path3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_inputs[2])
    path3 = BatchNormalization()(path3)

    # Concatenate outputs of Block 1
    block1_output = Concatenate()([path1, path2, path3])

    # Block 2
    path1_block2 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(block1_output)

    path2_block2 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(block1_output)
    path2_block2 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(path2_block2)

    path3_block2 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(block1_output)
    path3_sub1 = SeparableConv2D(filters=32, kernel_size=(1, 3), padding='same', activation='relu')(path3_block2)
    path3_sub2 = SeparableConv2D(filters=32, kernel_size=(3, 1), padding='same', activation='relu')(path3_block2)
    path3_output = Concatenate()([path3_sub1, path3_sub2])

    path4_block2 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(block1_output)
    path4_sub1 = SeparableConv2D(filters=32, kernel_size=(1, 3), padding='same', activation='relu')(path4_block2)
    path4_sub2 = SeparableConv2D(filters=32, kernel_size=(3, 1), padding='same', activation='relu')(path4_block2)
    path4_output = Concatenate()([path4_sub1, path4_sub2])
    
    # Concatenate outputs of Block 2
    block2_output = Concatenate()([path1_block2, path2_block2, path3_output, path4_output])

    # Final layers
    flatten_layer = Flatten()(block2_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model