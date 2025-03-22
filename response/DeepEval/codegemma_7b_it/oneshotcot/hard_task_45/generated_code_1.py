import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # First block
    split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split[0])
    path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split[1])
    path3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split[2])
    concat_path1_2_3 = Concatenate()([path1, path2, path3])

    # Second block
    path4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(concat_path1_2_3)
    path5 = Conv2D(filters=64, kernel_size=(1, 3), padding='same', activation='relu')(concat_path1_2_3)
    path6 = Conv2D(filters=64, kernel_size=(3, 1), padding='same', activation='relu')(concat_path1_2_3)
    path7 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(concat_path1_2_3)
    path8 = MaxPooling2D(pool_size=(2, 2), padding='same')(concat_path1_2_3)
    path9 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(path8)
    concat_path4_5_6_7_9 = Concatenate()([path4, path5, path6, path7, path9])

    # Output layer
    flatten_layer = Flatten()(concat_path4_5_6_7_9)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model