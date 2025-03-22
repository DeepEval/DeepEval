import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block - splitting the input and applying depthwise separable convolutions
    split_outputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Depthwise separable convolutions for the split outputs
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_outputs[0])
    conv1 = BatchNormalization()(conv1)

    conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_outputs[1])
    conv2 = BatchNormalization()(conv2)

    conv3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_outputs[2])
    conv3 = BatchNormalization()(conv3)

    # Concatenating the outputs from the first block
    block1_output = Concatenate()([conv1, conv2, conv3])

    # Second block - Multiple branches for feature extraction
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(block1_output)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch1)

    branch3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(block1_output)
    branch3 = Conv2D(filters=64, kernel_size=(1, 7), padding='same', activation='relu')(branch3)
    branch3 = Conv2D(filters=64, kernel_size=(7, 1), padding='same', activation='relu')(branch3)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch3)

    branch4 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(block1_output)

    # Concatenating the outputs from the second block
    block2_output = Concatenate()([branch2, branch3, branch4])

    # Flatten and Dense layers for classification
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model