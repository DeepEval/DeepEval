import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, SeparableConv2D, GlobalAveragePooling2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    split_input = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)
    path1 = Lambda(lambda x: SeparableConv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x))((split_input[0]))
    path2 = Lambda(lambda x: SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x))((split_input[1]))
    path3 = Lambda(lambda x: SeparableConv2D(filters=96, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x))((split_input[2]))
    concat1 = Concatenate()([path1, path2, path3])

    # Second block
    path4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat1)
    path5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path4)
    path6 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path5)
    path7 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(path6)

    # Concatenate outputs from both blocks
    concat2 = Concatenate()([path7, concat1])

    # Global average pooling and fully connected layer
    gap = GlobalAveragePooling2D()(concat2)
    output_layer = Dense(units=10, activation='softmax')(gap)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model