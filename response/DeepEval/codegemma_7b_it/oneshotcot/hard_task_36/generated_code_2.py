import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():

    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Main pathway
    conv_main = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_main_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_main)
    conv_main_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_main)
    max_pooling_main = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_main_2)
    dropout_main = Dropout(0.5)(max_pooling_main)

    # Branch pathway
    conv_branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling_branch = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_branch)

    # Fuse pathways
    concat = Concatenate()([dropout_main, max_pooling_branch])

    # Global average pooling
    global_avg_pooling = keras.layers.GlobalAveragePooling2D()(concat)

    # Flatten and fully connected layer
    flatten_layer = Flatten()(global_avg_pooling)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Model definition
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model