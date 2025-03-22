import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Dropout, Concatenate, Flatten, Dense, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Splitting the input and applying convolutions
    def block_1(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        concatenated = Concatenate()([conv1, conv2, conv3])
        dropout = Dropout(rate=0.5)(concatenated)
        return dropout

    # Block 2: Four branches with different operations
    def block_2(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(Conv2D(filters=64, kernel_size=(3, 3), padding='same')(input_tensor))
        path3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(Conv2D(filters=64, kernel_size=(5, 5), padding='same')(input_tensor))
        path4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor))
        concatenated = Concatenate()([path1, path2, path3, path4])
        return concatenated

    block1_output = block_1(input_layer)
    block2_output = block_2(block1_output)

    # Flattening and fully connected layers for output
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model