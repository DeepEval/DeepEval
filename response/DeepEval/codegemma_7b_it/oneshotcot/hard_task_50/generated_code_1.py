import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dense, Dropout, Concatenate, Reshape, Lambda
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.regularizers import l2

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # First block
    max_pooling_1x1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_layer)
    max_pooling_2x2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    max_pooling_4x4 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(input_layer)

    flattened_1x1 = Flatten()(max_pooling_1x1)
    flattened_2x2 = Flatten()(max_pooling_2x2)
    flattened_4x4 = Flatten()(max_pooling_4x4)

    dropout_layer = Dropout(0.25)(Concatenate()([flattened_1x1, flattened_2x2, flattened_4x4]))

    # Second block
    split_layer = Lambda(lambda x: tf.split(x, 4, axis=-1))(dropout_layer)

    path1 = Lambda(lambda x: x[:, :, :, 0:1])(split_layer)
    path2 = Lambda(lambda x: x[:, :, :, 1:2])(split_layer)
    path3 = Lambda(lambda x: x[:, :, :, 2:3])(split_layer)
    path4 = Lambda(lambda x: x[:, :, :, 3:4])(split_layer)

    conv_1x1 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', kernel_initializer=TruncatedNormal(stddev=0.02), kernel_regularizer=l2(1e-5))(x))(path1)
    conv_3x3 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer=TruncatedNormal(stddev=0.02), kernel_regularizer=l2(1e-5))(x))(path2)
    conv_5x5 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer=TruncatedNormal(stddev=0.02), kernel_regularizer=l2(1e-5))(x))(path3)
    conv_7x7 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu', kernel_initializer=TruncatedNormal(stddev=0.02), kernel_regularizer=l2(1e-5))(x))(path4)

    concat_layer = Concatenate()([conv_1x1, conv_3x3, conv_5x5, conv_7x7])

    reshape_layer = Reshape((concat_layer.shape[1], concat_layer.shape[2], concat_layer.shape[3]))(concat_layer)

    # Output layer
    output_layer = Flatten()(reshape_layer)
    output_layer = Dense(units=10, activation='softmax')(output_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model