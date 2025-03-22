import keras
from keras.layers import Input, Conv2D, SeparableConv2D, Dropout, Add, Lambda, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First Block
    # Main path
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    dropout1 = Dropout(rate=0.3)(conv1)
    conv2 = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout1)

    # Branch path (direct connection)
    branch_path = input_layer

    # Add main path and branch path
    block1_output = Add()([conv2, branch_path])

    # Second Block
    def block_2(input_tensor):
        # Split input into three groups
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Separable convolutions for each group
        conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        dropout_conv1 = Dropout(rate=0.3)(conv1)

        conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        dropout_conv2 = Dropout(rate=0.3)(conv2)

        conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        dropout_conv3 = Dropout(rate=0.3)(conv3)

        # Concatenate the outputs
        output_tensor = Concatenate()([dropout_conv1, dropout_conv2, dropout_conv3])
        return output_tensor

    block2_output = block_2(input_tensor=block1_output)

    # Flatten and fully connected layer for classification
    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model