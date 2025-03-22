import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Dropout, Add, Lambda, Flatten, Concatenate, DepthwiseConv2D, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    # Main Path
    conv_main = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    dropout_main = Dropout(rate=0.25)(conv_main)
    conv_restore = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout_main)

    # Branch Path
    branch_path = input_layer  # Directly connects to input

    # Adding the outputs of both paths
    block1_output = Add()([conv_restore, branch_path])

    # Block 2
    # Splitting the output into three groups
    split_output = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(block1_output)

    # Each group using Depthwise Separable Convolutions with different kernel sizes
    conv1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(split_output[0])
    dropout1 = Dropout(rate=0.25)(conv1)

    conv2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(split_output[1])
    dropout2 = Dropout(rate=0.25)(conv2)

    conv3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(split_output[2])
    dropout3 = Dropout(rate=0.25)(conv3)

    # Concatenating the outputs from all groups
    block2_output = Concatenate()([dropout1, dropout2, dropout3])

    # Final Layers
    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model