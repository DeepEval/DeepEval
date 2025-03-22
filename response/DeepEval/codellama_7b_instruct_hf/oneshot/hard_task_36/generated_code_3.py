import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Main pathway
    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
    maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3)
    dropout = Dropout(0.5)(maxpool)

    # Branch pathway
    branch_input = Input(shape=(28, 28, 1))
    branch_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(branch_input)
    branch_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch_conv)

    # Fusion
    fusion = Concatenate()([maxpool, branch_pool])

    # Flatten and fully connected layers
    flatten = Flatten()(fusion)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create and return model
    model = keras.Model(inputs=[input_layer, branch_input], outputs=output_layer)

    return model