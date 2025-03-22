import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def block(input_tensor):
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_initializer='he_normal', use_bias=False)(path1)
    path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)
    path4 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
    output_tensor = Concatenate()([path1, path2, path3, path4])

    return output_tensor

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)

    branch1_output = block(max_pooling)
    branch2_output = block(max_pooling)
    branch3_output = block(max_pooling)

    concat = Concatenate()([branch1_output, branch2_output, branch3_output])
    bath_norm = BatchNormalization()(concat)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model