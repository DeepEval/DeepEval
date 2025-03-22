import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Concatenate, AveragePooling2D

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # Main pathway
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_1)
    dropout1 = Dropout(0.5)(maxpool1)

    # Branch pathway
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    maxpool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

    # Fusion of main and branch pathways
    concat = Concatenate()([maxpool1, maxpool2])

    # Additional layers
    conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)
    conv4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)
    add_layer = Add()([conv3, conv4])
    avg_pool = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='same')(add_layer)
    flatten = Flatten()(avg_pool)
    dense = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model