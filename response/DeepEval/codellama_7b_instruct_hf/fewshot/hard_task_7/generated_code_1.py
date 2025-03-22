import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Concatenate, DepthwiseConv2D, Dense, Reshape

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    split1 = Lambda(lambda x: tf.split(value=x, num_or_size_splits=2, axis=-1))(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split1[0])
    depthwise = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split1[1])
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(depthwise)
    merge = Concatenate()([conv2, conv3])

    # Block 2
    input_layer = merge
    channels = input_layer.get_shape().as_list()[-1]
    groups = 4
    channels_per_group = channels // groups
    shape = (input_layer.shape[0], input_layer.shape[1], input_layer.shape[2], groups, channels_per_group)
    reshaped = Reshape(shape)(input_layer)
    reshaped = Reshape((-1, groups, channels_per_group))(reshaped)
    reshaped = Permute((0, 2, 1, 3))(reshaped)
    reshaped = Reshape((-1, channels_per_group, groups))(reshaped)
    flattened = Flatten()(reshaped)
    output_layer = Dense(units=10, activation='softmax')(flattened)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model