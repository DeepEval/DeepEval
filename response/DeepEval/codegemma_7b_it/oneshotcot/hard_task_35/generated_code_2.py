import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, Reshape

def block(input_tensor):
    GAP = GlobalAveragePooling2D()(input_tensor)
    dense1 = Dense(units=64, activation='relu')(GAP)
    dense2 = Dense(units=input_tensor.shape[-1], activation='sigmoid')(dense1)
    reshaped_weights = Reshape((1, 1, input_tensor.shape[-1]))(dense2)
    weighted_input = Multiply()([input_tensor, reshaped_weights])
    return weighted_input

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)

    branch1 = block(max_pooling)
    branch2 = block(max_pooling)

    concat = Concatenate()([branch1, branch2])
    bath_norm = BatchNormalization()(concat)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model