import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, Lambda, Add, Flatten, Dense


def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    groups = Lambda(function=lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)
    inputs_channels = input_layer.shape[-1]

    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(groups[0])
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)
    conv1_3 = Conv2D(filters=inputs_channels, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(conv1_2)
    group_0 = conv1_3

    conv2_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(groups[1])
    conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_1)
    conv2_3 = Conv2D(filters=inputs_channels, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(conv2_2)
    group_1 = conv2_3

    conv3_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(groups[2])
    conv3_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3_1)
    conv3_3 = Conv2D(filters=inputs_channels, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(conv3_2)
    group_2 = conv3_3
    main_path = Add()([group_0, group_1, group_2])

    added = Add()([main_path, input_layer])
    flatten = Flatten()(added)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

