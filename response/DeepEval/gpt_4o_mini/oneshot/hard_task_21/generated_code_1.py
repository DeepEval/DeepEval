import keras
import tensorflow as tf
from keras.layers import Input, Lambda, Conv2D, Concatenate, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel
    split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Main path with depthwise separable convolutions
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split[0])
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split[1])
    conv3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split[2])

    # Concatenate the outputs of the main path
    main_path_output = Concatenate()([conv1, conv2, conv3])

    # Branch path with a 1x1 convolution
    branch_path_output = Conv2D(filters=96, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Add the outputs of the main and branch paths
    combined_output = Add()([main_path_output, branch_path_output])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model