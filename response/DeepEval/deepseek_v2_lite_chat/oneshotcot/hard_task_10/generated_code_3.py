import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Feature extraction path 1: 1x1 convolution
    conv_path1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    # Feature extraction path 2: sequence of convolutions
    conv_path2 = Conv2D(filters=16, kernel_size=(1, 7), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    conv_path2 = Conv2D(filters=16, kernel_size=(7, 1), strides=(1, 1), padding='valid', activation='relu')(conv_path2)

    # Concatenate the outputs from the two paths
    concat_layer = Concatenate()([conv_path1, conv_path2])

    # Add a 1x1 convolution to align the output dimensions with the input image's channel
    conv_output = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(concat_layer)

    # Direct branch connection from input to the output of the main path
    branch = Input(shape=(32, 32, 3))
    branch_conv = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(branch_conv)
    branch_output = Add()([conv_output, branch_conv])

    # Classification results will be produced through two fully connected layers
    dense1 = Dense(units=128, activation='relu')(branch_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model([input_layer, branch_conv], [output_layer, branch_output])

    return model