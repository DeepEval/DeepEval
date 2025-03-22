import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from tensorflow import split

def dl_model():     

    input_layer = Input(shape=(32, 32, 3)) 

    # Split the input channels
    split_channels = Lambda(lambda x: split(x, num_or_size_splits=3, axis=2))(input_layer)

    # Process each channel group
    group1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_channels[0])
    group2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_channels[1])
    group3 = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_channels[2])

    # Concatenate the outputs
    merged = Concatenate(axis=2)([group1, group2, group3])

    # Flatten and add dense layers
    flatten_layer = Flatten()(merged)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model