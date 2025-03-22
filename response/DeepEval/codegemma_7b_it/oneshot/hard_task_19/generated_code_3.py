import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Add

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3)

    # Branch Path
    global_avg_pooling = GlobalAveragePooling2D()(input_layer)
    flatten_avg = Flatten()(global_avg_pooling)
    dense1_avg = Dense(units=128, activation='relu')(flatten_avg)
    dense2_avg = Dense(units=64, activation='relu')(dense1_avg)
    reshape_avg = Reshape((1, 64))(dense2_avg)

    dense1_max = Dense(units=128, activation='relu')(max_pooling)
    dense2_max = Dense(units=64, activation='relu')(dense1_max)
    reshape_max = Reshape((1, 64))(dense2_max)

    multiply = Multiply()([reshape_avg, reshape_max])
    concat = Add()([multiply, max_pooling])

    # Output Layers
    flatten_concat = Flatten()(concat)
    dense3 = Dense(units=10, activation='softmax')(flatten_concat)

    model = keras.Model(inputs=input_layer, outputs=dense3)

    return model