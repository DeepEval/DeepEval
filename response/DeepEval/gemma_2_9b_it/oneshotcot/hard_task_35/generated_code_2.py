import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Branch 1
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch1)

    def attention_block(input_tensor):
        pooled = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=input_tensor.shape[-1], activation='relu')(pooled)
        dense2 = Dense(units=input_tensor.shape[-1], activation='relu')(dense1)
        attention_weights = Reshape((input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]))(dense2)
        return input_tensor * attention_weights

    branch1 = attention_block(branch1)

    # Branch 2
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch2)

    branch2 = attention_block(branch2)

    # Concatenate branches
    merged = Concatenate()([branch1, branch2])

    # Flatten and fully connected
    flatten_layer = Flatten()(merged)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model