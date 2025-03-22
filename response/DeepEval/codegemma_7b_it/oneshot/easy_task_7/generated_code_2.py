import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))

    # Main Path
    conv_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    dropout_1 = Dropout(rate=0.25)(conv_1)
    conv_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout_1)
    dropout_2 = Dropout(rate=0.25)(conv_2)
    conv_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout_2)
    conv_4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_3)
    restore_channel = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_4)

    # Branch Path
    branch_conv_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_conv_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_conv_1)

    # Combine Main and Branch Paths
    combined = keras.layers.Add()([restore_channel, branch_conv_2])

    # Additional Processing
    bath_norm = BatchNormalization()(combined)
    flatten_layer = Flatten()(bath_norm)
    dense_1 = Dense(units=128, activation='relu')(flatten_layer)
    dense_2 = Dense(units=10, activation='softmax')(dense_1)

    model = keras.Model(inputs=input_layer, outputs=dense_2)

    return model