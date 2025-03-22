import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    conv_initial = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 1
    conv_branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_initial)

    # Branch 2
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_initial)
    conv_branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling2)

    # Branch 3
    max_pooling3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_initial)
    conv_branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling3)
    upsample3 = UpSampling2D(size=(2, 2))(conv_branch3)

    # Branch 4
    max_pooling4 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_initial)
    conv_branch4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling4)
    upsample4 = UpSampling2D(size=(4, 4))(conv_branch4)

    # Concatenate all branches
    concat = Concatenate()([conv_branch1, conv_branch2, upsample3, upsample4])

    # Final convolutional layer
    conv_final = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)

    # Flatten and fully connected layers
    flatten = Flatten()(conv_final)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

model = dl_model()