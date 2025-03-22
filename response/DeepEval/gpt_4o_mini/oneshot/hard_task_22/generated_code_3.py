import keras
import tensorflow as tf
from keras.layers import Input, Lambda, SeparableConv2D, Concatenate, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel
    split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Main path
    main_path_outputs = []
    kernel_sizes = [1, 3, 5]
    
    for i, kernel_size in enumerate(kernel_sizes):
        sep_conv = SeparableConv2D(filters=32, kernel_size=(kernel_size, kernel_size), padding='same', activation='relu')(split_inputs[i])
        main_path_outputs.append(sep_conv)

    # Concatenate the outputs of the main path
    main_path_output = Concatenate()(main_path_outputs)

    # Branch path
    branch_output = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Fuse the outputs of both paths
    combined_output = Add()([main_path_output, branch_output])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model