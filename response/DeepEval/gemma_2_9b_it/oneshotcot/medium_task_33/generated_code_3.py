import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from tensorflow import split

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Split the channels into three groups
    def split_channels(tensor):
        r, g, b = split(tensor, num_or_size_splits=3, axis=2)
        return [r, g, b]

    split_tensors = Lambda(split_channels)(input_layer)

    # Feature extraction for each channel group
    group1_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensors[0])
    group2_output = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_tensors[1])
    group3_output = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_tensors[2])

    # Concatenate outputs
    merged_tensor = Concatenate(axis=2)([group1_output, group2_output, group3_output])

    # Fully connected layers
    flatten_layer = Flatten()(merged_tensor)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model