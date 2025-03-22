import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Flatten, Concatenate, Reshape, Multiply

def block(input_tensor):
    gap = GlobalAveragePooling2D()(input_tensor)
    fc1 = Dense(units=input_tensor.shape[-1], activation='relu')(gap)
    fc2 = Dense(units=input_tensor.shape[-1], activation='sigmoid')(fc1)
    reshaped_weights = Reshape((1, 1, input_tensor.shape[-1]))(fc2)
    output_tensor = Multiply()([reshaped_weights, input_tensor])
    return output_tensor

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)

    branch_1 = block(max_pooling)
    branch_2 = block(max_pooling)

    concat = Concatenate()([branch_1, branch_2])

    flatten_layer = Flatten()(concat)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model