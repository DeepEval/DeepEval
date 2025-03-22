import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Concatenate, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Feature Extraction
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_1)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_2)
    conv1_3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_3)

    # Feature Enhancement
    conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_2)
    drop2 = Dropout(rate=0.3)(conv2_1)
    conv2_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(drop2)

    # Upsampling and Skip Connections
    upsample3 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv2_2)
    concat3 = Concatenate()([upsample3, conv1_3])
    conv3_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat3)
    upsample2 = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv3_1)
    concat2 = Concatenate()([upsample2, conv1_2])
    conv2_3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat2)
    upsample1 = Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv2_3)
    concat1 = Concatenate()([upsample1, conv1_1])
    conv1_4 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat1)

    # Classification
    flatten = keras.layers.Flatten()(conv1_4)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model