import keras
from keras.layers import Input, SeparableConv2D, BatchNormalization, Concatenate, Flatten, Dense, Lambda
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Block 1
    split_tensors = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)

    # Three separate paths with different kernel sizes
    path1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_tensors[0])
    path1 = BatchNormalization()(path1)

    path2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_tensors[1])
    path2 = BatchNormalization()(path2)

    path3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_tensors[2])
    path3 = BatchNormalization()(path3)

    # Concatenate the outputs of block 1
    block1_output = Concatenate()([path1, path2, path3])

    # Block 2
    # Path 1
    path1_block2 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(block1_output)

    # Path 2
    path2_block2 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(block1_output)
    path2_block2 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(path2_block2)

    # Path 3
    path3_block2 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(block1_output)
    sub_path1 = SeparableConv2D(filters=32, kernel_size=(1, 3), padding='same', activation='relu')(path3_block2)
    sub_path2 = SeparableConv2D(filters=32, kernel_size=(3, 1), padding='same', activation='relu')(path3_block2)
    path3_block2 = Concatenate()([sub_path1, sub_path2])

    # Path 4
    path4_block2 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(block1_output)
    path4_block2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(path4_block2)
    sub_path1 = SeparableConv2D(filters=32, kernel_size=(1, 3), padding='same', activation='relu')(path4_block2)
    sub_path2 = SeparableConv2D(filters=32, kernel_size=(3, 1), padding='same', activation='relu')(path4_block2)
    path4_block2 = Concatenate()([sub_path1, sub_path2])

    # Concatenate the outputs of block 2
    block2_output = Concatenate()([path1_block2, path2_block2, path3_block2, path4_block2])

    # Flattening and Dense layers for classification
    flatten_layer = Flatten()(block2_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)  # CIFAR-10 has 10 classes

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model