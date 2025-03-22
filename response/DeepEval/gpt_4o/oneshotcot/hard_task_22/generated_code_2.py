import tensorflow as tf
from keras.layers import Input, Lambda, SeparableConv2D, Concatenate, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Splitting the input into three groups along the channel axis
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Main Path: Multi-scale feature extraction
    path1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_layer[0])
    path2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_layer[1])
    path3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_layer[2])

    main_path_output = Concatenate()([path1, path2, path3])

    # Branch Path: 1x1 convolution to align the number of channels
    branch_path_output = SeparableConv2D(filters=96, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Fusion by adding the outputs from the main path and the branch path
    fused_output = Add()([main_path_output, branch_path_output])

    # Flatten and fully connected layers for classification
    flatten_layer = Flatten()(fused_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model