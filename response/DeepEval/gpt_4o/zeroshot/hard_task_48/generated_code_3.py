from tensorflow.keras.layers import Input, Lambda, SeparableConv2D, BatchNormalization, Concatenate, Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model
import tensorflow as tf

def dl_model():
    # Define input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Splitting and using separable convolutions
    def split_input(x):
        return tf.split(x, num_or_size_splits=3, axis=-1)

    split_layer = Lambda(split_input)(input_layer)

    # SeparableConv2D branches with BatchNormalization
    conv_1x1 = SeparableConv2D(32, (1, 1), padding='same', activation='relu')(split_layer[0])
    conv_1x1 = BatchNormalization()(conv_1x1)

    conv_3x3 = SeparableConv2D(32, (3, 3), padding='same', activation='relu')(split_layer[1])
    conv_3x3 = BatchNormalization()(conv_3x3)

    conv_5x5 = SeparableConv2D(32, (5, 5), padding='same', activation='relu')(split_layer[2])
    conv_5x5 = BatchNormalization()(conv_5x5)

    # Concatenate the outputs of separable convolutions
    block1_output = Concatenate()([conv_1x1, conv_3x3, conv_5x5])

    # Block 2: Creating the four parallel paths
    # Path 1: 1x1 Convolution
    path1 = Conv2D(64, (1, 1), padding='same', activation='relu')(block1_output)

    # Path 2: 3x3 Average Pooling followed by 1x1 Convolution
    path2 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(block1_output)
    path2 = Conv2D(64, (1, 1), padding='same', activation='relu')(path2)

    # Path 3: 1x1 Convolution, then splits into 1x3 and 3x1
    path3 = Conv2D(64, (1, 1), padding='same', activation='relu')(block1_output)
    path3_1x3 = Conv2D(64, (1, 3), padding='same', activation='relu')(path3)
    path3_3x1 = Conv2D(64, (3, 1), padding='same', activation='relu')(path3)
    path3 = Concatenate()([path3_1x3, path3_3x1])

    # Path 4: 1x1 Convolution, 3x3 Convolution, then splits into 1x3 and 3x1
    path4 = Conv2D(64, (1, 1), padding='same', activation='relu')(block1_output)
    path4 = Conv2D(64, (3, 3), padding='same', activation='relu')(path4)
    path4_1x3 = Conv2D(64, (1, 3), padding='same', activation='relu')(path4)
    path4_3x1 = Conv2D(64, (3, 1), padding='same', activation='relu')(path4)
    path4 = Concatenate()([path4_1x3, path4_3x1])

    # Concatenate all paths
    block2_output = Concatenate()([path1, path2, path3, path4])

    # Flatten and Dense layer for final classification
    flatten = Flatten()(block2_output)
    output_layer = Dense(10, activation='softmax')(flatten)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model