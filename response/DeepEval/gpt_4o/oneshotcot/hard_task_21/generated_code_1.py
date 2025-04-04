import keras
from keras.layers import Input, Lambda, SeparableConv2D, Conv2D, Concatenate, Add, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    # Split the input into 3 groups along the channel axis
    split_channels = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Apply depthwise separable convolutions with different kernel sizes
    group1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_channels[0])
    group2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_channels[1])
    group3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_channels[2])

    # Concatenate the outputs
    main_path_output = Concatenate()([group1, group2, group3])

    # Branch path
    branch_path_output = Conv2D(filters=96, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Add outputs from main path and branch path
    combined_output = Add()([main_path_output, branch_path_output])

    # Flatten and Dense layers for classification
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model