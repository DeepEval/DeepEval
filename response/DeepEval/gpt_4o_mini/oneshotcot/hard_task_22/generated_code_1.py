import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Concatenate, Flatten, Dense, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path: Split the input into three groups along the channel
    split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Multi-scale feature extraction using separable convolutions
    path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_inputs[0])
    path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_inputs[1])
    path3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_inputs[2])

    # Concatenate the outputs from the main path
    main_output = Concatenate()([path1, path2, path3])

    # Branch path: Apply a 1x1 convolution to the input
    branch_output = Conv2D(filters=96, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Fuse both paths through addition
    combined_output = Add()([main_output, branch_output])

    # Flatten the output and pass through fully connected layers
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model