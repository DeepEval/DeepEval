import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     
    input_layer = Input(shape=(28, 28, 1))

    # Parallel branch 1
    x1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x1_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x1)
    x1_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x1_1)

    # Parallel branch 2
    x2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x2_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x2)
    x2_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x2_1)

    # Concatenate outputs of parallel branches
    merged_output = Concatenate()([x1_2, x2_2])

    # Flatten and fully connected layer
    flattened = Flatten()(merged_output)
    output_layer = Dense(units=10, activation='softmax')(flattened)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model