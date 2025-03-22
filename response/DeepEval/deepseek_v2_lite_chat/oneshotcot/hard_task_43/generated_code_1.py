import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    avg_pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(conv1)
    avg_pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
    avg_pool3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(conv1)
    flat_output1 = Flatten()(avg_pool3)
    flat_output2 = Flatten()(avg_pool2)
    flat_output = Concatenate()([flat_output1, flat_output2])

    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(flat_output)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(flat_output)
    conv4 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(flat_output)
    avg_pool4 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(flat_output)
    avg_pool5 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(flat_output)
    avg_pool6 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(flat_output)
    flat_output3 = Flatten()(avg_pool6)
    flat_output4 = Flatten()(avg_pool5)
    flat_output5 = Concatenate()([flat_output3, flat_output4])

    conv5 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(flat_output5)
    conv6 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(flat_output5)
    conv7 = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(flat_output5)
    avg_pool7 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(flat_output5)
    avg_pool8 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(flat_output5)
    avg_pool9 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(flat_output5)
    flat_output6 = Flatten()(avg_pool9)
    flat_output7 = Flatten()(avg_pool8)
    flat_output8 = Concatenate()([flat_output6, flat_output7])

    conv8 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(flat_output8)
    conv9 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(flat_output8)
    conv10 = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(flat_output8)
    avg_pool10 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(flat_output8)
    avg_pool11 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(flat_output8)
    avg_pool12 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(flat_output8)
    flat_output9 = Flatten()(avg_pool12)
    flat_output10 = Flatten()(avg_pool11)
    flat_output11 = Concatenate()([flat_output9, flat_output10])

    dense1 = Dense(units=512, activation='relu')(flat_output11)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model