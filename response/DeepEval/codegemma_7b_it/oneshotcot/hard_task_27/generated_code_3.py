import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.initializers import RandomNormal
from keras.regularizers import l2

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Spatial feature extraction with depthwise separable convolutional layer
    conv = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same', use_bias=False,
                 kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(input_layer)
    bn = BatchNormalization(axis=3)(conv)
    act = keras.activations.relu(bn)
    mp = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(act)

    # Channel-wise feature transformation with fully connected layers
    fc1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False,
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(mp)
    bn_fc1 = BatchNormalization(axis=3)(fc1)
    act_fc1 = keras.activations.relu(bn_fc1)
    fc2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False,
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(act_fc1)
    bn_fc2 = BatchNormalization(axis=3)(fc2)
    act_fc2 = keras.activations.relu(bn_fc2)

    # Combine original input with processed features
    output = keras.layers.Add()([act_fc2, input_layer])

    # Output layer for classification
    avg_pool = keras.layers.AveragePooling2D()(output)
    flat = Flatten()(avg_pool)
    fc3 = Dense(units=64, activation='relu', kernel_regularizer=l2(0.001))(flat)
    output_layer = Dense(units=10, activation='softmax')(fc3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model