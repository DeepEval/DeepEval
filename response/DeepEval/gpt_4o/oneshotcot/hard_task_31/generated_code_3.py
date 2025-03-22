import keras
from keras.layers import Input, Conv2D, SeparableConv2D, Dropout, Lambda, Add, Flatten, Dense, Concatenate
import tensorflow as tf

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # First Block
    # Main Path
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    dropout1 = Dropout(rate=0.3)(conv1)
    conv2 = Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation='relu')(dropout1)

    # Branch Path
    branch = input_layer

    # Adding the Main Path and Branch Path
    block1_output = Add()([conv2, branch])

    # Second Block
    def group_split(input_tensor):
        # Split the input into three groups along the last dimension
        return tf.split(input_tensor, num_or_size_splits=3, axis=-1)

    # Using Lambda to split
    split_layer = Lambda(group_split)(block1_output)

    # Group 1 - Separable Convolution 1x1
    sep_conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_layer[0])
    dropout2 = Dropout(rate=0.3)(sep_conv1)

    # Group 2 - Separable Convolution 3x3
    sep_conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_layer[1])
    dropout3 = Dropout(rate=0.3)(sep_conv2)

    # Group 3 - Separable Convolution 5x5
    sep_conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_layer[2])
    dropout4 = Dropout(rate=0.3)(sep_conv3)

    # Concatenate the outputs of the three groups
    concat_layer = Concatenate()([dropout2, dropout3, dropout4])

    # Flatten and output layer
    flatten_layer = Flatten()(concat_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model