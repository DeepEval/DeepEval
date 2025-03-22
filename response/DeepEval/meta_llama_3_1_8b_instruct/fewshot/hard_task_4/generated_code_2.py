import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Flatten, Dense, Reshape, Multiply
from keras.layers import Concatenate, Activation

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    conv1 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    attention_weights = GlobalAveragePooling2D()(conv2)
    attention_weights = Dense(units=3, activation='relu')(attention_weights)
    attention_weights = Dense(units=3, activation='sigmoid')(attention_weights)
    attention_weights = Reshape(target_shape=(1, 1, 3))(attention_weights)
    attention_weights = Multiply()([conv2, attention_weights])

    conv3 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(attention_weights)
    adding_layer = Concatenate()([input_layer, conv3])

    flatten_layer = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model