import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))

    # Main path
    conv1_1x1_group1 = layers.Conv2D(64, (1, 1), padding='same', use_bias=False)(inputs)
    conv1_1x1_group2 = layers.Conv2D(64, (1, 1), padding='same', use_bias=False)(inputs)
    conv1_1x1_group3 = layers.Conv2D(64, (1, 1), padding='same', use_bias=False)(inputs)

    # Concatenate groups along channel dimension
    concat_group = layers.Concatenate(axis=-1)([conv1_1x1_group1, conv1_1x1_group2, conv1_1x1_group3])

    # Block 2
    conv2_1x1_group1 = layers.Conv2D(64, (1, 1), padding='same', use_bias=False)(concat_group)
    conv2_1x1_group2 = layers.Conv2D(64, (1, 1), padding='same', use_bias=False)(concat_group)
    conv2_1x1_group3 = layers.Conv2D(64, (1, 1), padding='same', use_bias=False)(concat_group)

    # Reshape and channel shuffle
    shape_before_reshape = tf.shape(conv2_1x1_group1)
    reshape_group1 = layers.Reshape((shape_before_reshape[1], shape_before_reshape[2], 3, 64//3))(conv2_1x1_group1)
    reshape_group2 = layers.Reshape((shape_before_reshape[1], shape_before_reshape[2], 3, 64//3))(conv2_1x1_group2)
    reshape_group3 = layers.Reshape((shape_before_reshape[1], shape_before_reshape[2], 3, 64//3))(conv2_1x1_group3)

    permute_group1 = layers.Permute((1, 2, 4, 3))(reshape_group1)
    permute_group2 = layers.Permute((1, 2, 4, 3))(reshape_group2)
    permute_group3 = layers.Permute((1, 2, 4, 3))(reshape_group3)

    concat_group_reshaped = layers.Concatenate(axis=-1)([permute_group1, permute_group2, permute_group3])

    # Block 3
    conv3_3x3_dw = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', use_bias=False)(concat_group_reshaped)
    conv3_3x3_dw = layers.BatchNormalization()(conv3_3x3_dw)
    conv3_3x3_pw = layers.Conv2D(64, (1, 1), strides=1, padding='same', use_bias=False)(conv3_3x3_dw)
    conv3_3x3_pw = layers.BatchNormalization()(conv3_3x3_pw)

    # Branch
    branch_conv1_1x1 = layers.Conv2D(64, (1, 1), padding='same', use_bias=False)(inputs)
    branch_conv1_1x1 = layers.BatchNormalization()(branch_conv1_1x1)

    # Output
    concat_output = layers.Add()([branch_conv1_1x1, conv3_3x3_pw])
    concat_output = layers.BatchNormalization()(concat_output)

    # Classification head
    flatten = layers.Flatten()(concat_output)
    outputs = layers.Dense(10, activation='softmax')(flatten)

    # Model definition
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model