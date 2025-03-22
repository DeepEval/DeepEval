import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    
    # Main path
    conv_main1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    separable_conv_main1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_initializer='he_normal', depthwise_constraint=keras.initializers.non_neg(), use_bias=False)(conv_main1)
    max_pooling_main1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(separable_conv_main1)

    conv_main2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling_main1)
    separable_conv_main2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_initializer='he_normal', depthwise_constraint=keras.initializers.non_neg(), use_bias=False)(conv_main2)
    max_pooling_main2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(separable_conv_main2)

    # Branch path
    conv_branch = Conv2D(filters=64, kernel_size=(1, 1), strides=(2, 2), padding='valid', activation='relu')(input_layer)

    # Sum the outputs from both paths
    output = Concatenate()([max_pooling_main2, conv_branch])

    # Flatten and fully connected layer
    bath_norm = BatchNormalization()(output)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model