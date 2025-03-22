import keras
from keras.layers import Input, Lambda, Concatenate, MaxPooling2D, SeparableConv2D, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    split_layer = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    conv1 = SeparableConv2D(32, (1, 1), activation='relu')(split_layer[0])
    conv2 = SeparableConv2D(32, (3, 3), activation='relu')(split_layer[1])
    conv3 = SeparableConv2D(32, (5, 5), activation='relu')(split_layer[2])
    concat_layer = Concatenate()([conv1, conv2, conv3])
    pooling_layer = MaxPooling2D(pool_size=(2, 2), strides=2)(concat_layer)
    batch_norm_layer = BatchNormalization()(pooling_layer)
    flatten_layer = Flatten()(batch_norm_layer)
    dense1 = Dense(64, activation='relu')(flatten_layer)
    dense2 = Dense(32, activation='relu')(dense1)
    dense3 = Dense(10, activation='softmax')(dense2)
    model = keras.Model(inputs=input_layer, outputs=dense3)
    return model