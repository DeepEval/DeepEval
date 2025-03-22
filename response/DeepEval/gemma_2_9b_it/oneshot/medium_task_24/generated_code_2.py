import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Dropout, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3)) 

    # Branch 1
    conv1_1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)
    dropout1 = Dropout(0.25)(conv1_2)

    # Branch 2
    conv2_1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2_2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(conv2_1)
    conv2_3 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv2_2)
    conv2_4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_3)
    dropout2 = Dropout(0.25)(conv2_4)

    # Branch 3
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)

    # Concatenate branches
    merged = Concatenate()([dropout1, dropout2, max_pool])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(merged)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model