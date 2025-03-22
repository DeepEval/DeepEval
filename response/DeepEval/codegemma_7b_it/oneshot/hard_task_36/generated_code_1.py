import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main pathway
    conv_main = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    conv_main = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_main)
    max_pooling_main = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_main)
    dropout_main = Dropout(0.5)(max_pooling_main)

    # Branch pathway
    conv_branch = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(max_pooling_main)

    # Fusion layer
    concat_layers = Concatenate()([dropout_main, conv_branch])

    # Global average pooling
    global_avg_pool = GlobalAveragePooling2D()(concat_layers)

    # Flatten and fully connected layers
    flatten_layer = Flatten()(global_avg_pool)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model