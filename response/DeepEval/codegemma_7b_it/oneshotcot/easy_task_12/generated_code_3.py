import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # Main Path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    separable_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_initializer='he_uniform')(conv1)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(separable_conv1)

    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling1)
    separable_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_initializer='he_uniform')(conv2)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(separable_conv2)

    # Branch Path
    branch_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(2, 2), padding='valid', activation='relu')(max_pooling1)

    # Concatenation and Summation
    concat = Concatenate()([max_pooling2, branch_conv])
    summed = keras.layers.Add()([concat, branch_conv])

    # Flattening and Fully Connected Layers
    flatten_layer = Flatten()(summed)
    dense = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model