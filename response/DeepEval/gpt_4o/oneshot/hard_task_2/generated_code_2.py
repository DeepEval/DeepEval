import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense, Lambda
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input along the channel dimension into three groups
    split_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    def process_group(group):
        conv1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
        return conv3

    # Process each group with the series of convolutions
    processed_groups = [process_group(group) for group in split_groups]

    # Combine the outputs of the processed groups using addition
    main_path = Add()(processed_groups)

    # Fuse the main path with the original input using addition
    fused_output = Add()([main_path, input_layer])

    # Flatten the output and pass it through a fully connected layer
    flatten_layer = Flatten()(fused_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model