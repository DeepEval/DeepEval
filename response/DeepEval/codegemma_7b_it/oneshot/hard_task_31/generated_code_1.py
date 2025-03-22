import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Lambda, tf
from keras.models import Model

def dl_model():

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First block (main path and branch path)
    conv_main = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    dropout_main = Dropout(rate=0.25)(conv_main)
    conv_main = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(dropout_main)

    conv_branch = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    # Concatenate outputs from main path and branch path
    concat_path = Concatenate()([conv_main, conv_branch])

    # Second block (separable convolutions)
    split_layer = Lambda(lambda x: tf.split(x, 3, axis=-1))(concat_path)

    conv_1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
    dropout_1x1 = Dropout(rate=0.25)(conv_1x1)

    conv_3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_layer[1])
    dropout_3x3 = Dropout(rate=0.25)(conv_3x3)

    conv_5x5 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_layer[2])
    dropout_5x5 = Dropout(rate=0.25)(conv_5x5)

    concat_conv = Concatenate()([conv_1x1, dropout_3x3, dropout_5x5])

    # Flatten and fully connected layer
    flatten_layer = Flatten()(concat_conv)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model