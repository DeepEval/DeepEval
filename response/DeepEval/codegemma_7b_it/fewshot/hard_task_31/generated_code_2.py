import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Lambda, Flatten, Concatenate, DepthwiseConv2D, Dense, Reshape

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # First block
    def block_1(input_tensor):
        # Main path
        conv1_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)
        conv1_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_2)
        conv1_4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_3)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1_4)

        # Branch path
        conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_1)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2_2)

        # Add outputs from both paths
        adding_layer = keras.layers.Add()([pool1, pool2])
        output_tensor = adding_layer

        return output_tensor

    # Second block
    def block_2(input_tensor):
        # Split the input into three groups
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)

        # Process each group with separable convolutional layers
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv1 = Dropout(0.2)(conv1)
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv2 = Dropout(0.2)(conv2)
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        conv3 = Dropout(0.2)(conv3)

        # Concatenate the outputs from the three groups
        output_tensor = Concatenate()([conv1, conv2, conv3])

        return output_tensor

    # Construct the model
    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)

    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model