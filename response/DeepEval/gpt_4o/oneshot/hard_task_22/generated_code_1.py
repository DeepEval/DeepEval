import keras
from keras.layers import Input, Lambda, SeparableConv2D, Conv2D, Add, Flatten, Dense, Concatenate
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path: splitting input into three groups along the channel dimension
    split_channels = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Multi-scale feature extraction with separable convolutions
    path1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_channels[0])
    path2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_channels[1])
    path3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_channels[2])
    
    # Concatenate outputs of the main path
    main_path_output = Concatenate()([path1, path2, path3])

    # Branch path: 1x1 convolution to align output channels
    branch_path_output = Conv2D(filters=96, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Fuse both paths through addition
    combined_output = Add()([main_path_output, branch_path_output])

    # Flatten the result and pass through fully connected layers
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model