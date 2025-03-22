import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Dense, Concatenate, DepthwiseConv2D, Reshape
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)
    main_path = Concatenate()(main_path)
    main_path = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(main_path)
    main_path = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(main_path)
    main_path = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(main_path)
    main_path = Flatten()(main_path)
    main_path = Dense(units=64, activation='relu')(main_path)
    main_path = Dense(units=10, activation='softmax')(main_path)

    # Branch path
    branch_path = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)
    branch_path = Concatenate()(branch_path)
    branch_path = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_path)
    branch_path = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(branch_path)
    branch_path = Flatten()(branch_path)
    branch_path = Dense(units=64, activation='relu')(branch_path)
    branch_path = Dense(units=10, activation='softmax')(branch_path)

    # Model
    model = Model(inputs=input_layer, outputs=main_path)
    model.summary()

    return model