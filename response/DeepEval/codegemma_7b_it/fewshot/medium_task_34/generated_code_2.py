import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Concatenate, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Part 1: Feature Extraction
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool1)
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    # Part 2: Enhancement
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool2)
    dropout = Dropout(rate=0.5)(conv3)
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout)

    # Part 3: Upsampling
    upconv5 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='valid')(conv4)
    concat5 = Concatenate()([upconv5, conv2])
    upconv6 = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='valid')(concat5)
    concat6 = Concatenate()([upconv6, conv1])

    # Classification
    conv7 = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='softmax')(concat6)

    model = keras.Model(inputs=input_layer, outputs=conv7)

    return model