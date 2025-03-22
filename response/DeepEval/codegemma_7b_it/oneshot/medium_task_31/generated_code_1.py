import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, tf

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split[0])
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split[1])
    conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split[2])
    maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)

    output_tensor = Concatenate()([conv1, conv2, conv3, maxpool])

    bath_norm = BatchNormalization()(output_tensor)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model