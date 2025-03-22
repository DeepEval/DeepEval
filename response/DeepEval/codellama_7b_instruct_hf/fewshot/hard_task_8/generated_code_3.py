import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, DepthwiseConv2D, Flatten, Dense, Permute, Reshape, Concatenate

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    output_path = Concatenate()([conv2, branch_path])

    # Block 2
    features_shape = tf.shape(output_path)
    output_shape = (features_shape[1], features_shape[2], 4, 8)
    output_path = Permute((3, 1, 2, 4))(output_path)
    output_path = Reshape(target_shape=output_shape)(output_path)
    output_path = Concatenate()([output_path, output_path, output_path, output_path])
    output_path = Permute((1, 2, 3, 4))(output_path)
    output_path = Reshape(target_shape=(28, 28, 64))(output_path)

    # Classification
    output_path = Flatten()(output_path)
    output_path = Dense(units=128, activation='relu')(output_path)
    output_path = Dense(units=10, activation='softmax')(output_path)

    model = keras.Model(inputs=input_layer, outputs=output_path)

    return model