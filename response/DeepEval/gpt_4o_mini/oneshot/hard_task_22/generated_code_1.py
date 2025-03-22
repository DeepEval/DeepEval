import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Flatten, Dense, Add
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    split_layers = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Main path with separable convolutions
    path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_layers[0])
    path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_layers[1])
    path3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_layers[2])

    # Concatenate the outputs of the main path
    main_path_output = Concatenate()([path1, path2, path3])

    # Branch path with 1x1 convolution
    branch_path_output = Conv2D(filters=96, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Fuse the main path and branch path outputs
    fused_output = Add()([main_path_output, branch_path_output])

    # Flatten the output and apply fully connected layers
    flatten_layer = Flatten()(fused_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model