import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    group_size = 3
    groups = tf.split(input_layer, num_or_size_splits=group_size, axis=3)

    # Apply 1x1 convolutions to each group
    conv_outputs = []
    for group in groups:
        conv = Conv2D(filters=int(input_layer.shape[-1] // group_size), kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group)
        conv_outputs.append(conv)

    # Concatenate the outputs of the convolutions
    concat_output = Concatenate(axis=3)(conv_outputs)

    # Downsample each group using average pooling
    pool_outputs = []
    for conv_output in conv_outputs:
        pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_output)
        pool_outputs.append(pool)

    # Concatenate the outputs of the pooling layers
    concat_pool = Concatenate(axis=3)(pool_outputs)

    # Flatten the concatenated feature maps
    flatten_layer = Flatten()(concat_pool)

    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model