import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, Lambda, Add, Dropout, Concatenate, Dense


def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    groups = Lambda(function=lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)
    
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding="same", activation='relu')(groups[0])
    conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(conv1_1)
    group_0 = Dropout(rate=0.5)(conv1_2)
    
    conv2_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding="same", activation='relu')(groups[1])
    conv2_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(conv2_1)
    group_1 = Dropout(rate=0.5)(conv2_2)

    conv3_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding="same", activation='relu')(groups[2])
    conv3_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(conv3_1)
    group_2 = Dropout(rate=0.5)(conv3_2)

    main_path = Concatenate()([group_0, group_1, group_2])
    branch_path = Conv2D(filters = main_path.shape[-1], kernel_size=(1, 1), strides=(1, 1), padding="valid")(input_layer)
    added = Add()([main_path, branch_path])
    output_layer = Dense(units=10, activation='softmax')(added)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

