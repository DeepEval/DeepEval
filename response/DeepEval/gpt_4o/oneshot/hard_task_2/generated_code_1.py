import keras
from keras.layers import Input, Conv2D, Lambda, Flatten, Dense, Add
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    split_channels = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    def conv_series(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(conv2)
        return conv3

    # Process each split through the series of convolutions
    group1 = conv_series(split_channels[0])
    group2 = conv_series(split_channels[1])
    group3 = conv_series(split_channels[2])

    # Combine the outputs of the three groups using addition
    main_path = Add()([group1, group2, group3])

    # Fuse with the original input layer
    fused_output = Add()([main_path, input_layer])

    # Flatten the combined features and feed into a fully connected layer
    flatten_layer = Flatten()(fused_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model